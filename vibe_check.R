library(tidyverse)
library(rcanvas)

library(ollamar)
library(jsonlite)
library(glue)
library(tictoc)

library(tinytable)

# LLM parameters ----
model = 'gemma3:12b'
num_ctx = 8000

# Retrieve responses ----
## Canvas parameters ----
## STE Fall 2025
## <https://catcourses.ucmerced.edu/courses/35976>
course_id = '35976'
## Vibe check for Week 02
assignment_id = '503498'

options(.rcanvas.show.url = TRUE)
## API token: <https://github.com/daranzolin/rcanvas?tab=readme-ov-file#setup>
set_canvas_domain('https://catcourses.ucmerced.edu')

## Get students ----
students_df = get_course_items(course_id, item = "students") |>
      filter(!is.na(sis_user_id)) |>
      distinct() |>
      as_tibble() |>
      select(id, name, sortable_name) |>
      arrange(sortable_name) |>
      mutate(
            idx = {
                  row_number() %>%
                        str_pad(., pad = '0', width = max(str_length(.)))
            },
            prefix = glue('{idx}-{sortable_name}')
      )

## Get submissions ----
resps = get_submissions(course_id, 'assignments', assignment_id) |>
      select(id = user_id, body) |>
      filter(!is.na(body)) |>
      mutate(body = {
            body |> xfun::strip_html() |> textutils::HTMLdecode()
      }) |>
      left_join(students_df, by = 'id') |>
      select(student = prefix, body)

## Claude-generated responses for development
# resps = 'gen_responses.txt' |>
#       read_file() |>
#       str_split_1('==========')

# Parse responses ----
## LLM response schema ----
schema = list(
      type = 'object',
      properties = list(
            full_response = list(type = 'string'),
            quote = list(type = 'string'),
            question = list(type = 'string'),
            answer = list(type = 'string')
      ),
      required = list('full_response', 'quote', 'question', 'answer')
)

## Assignment instructions ----
instructions = '/Users/danhicks/Google Drive/Teaching/*STE/site/assignments/vibe_check.md' |>
      read_file()

## Task prompt ----
system = glue(
      'A college student was given a short, structured reading-reflection assignment. Your job is to identify the parts of the student\'s response, and arrange them into a JSON structure for further processing. 

Pay attention to the structure of the response, not the content. Do not address the questions or provide any commentary on the quote or their answer. 

The `full_response` field should contain the response exactly as given. For the other fields, if the response includes section numbers or paragraph headers like "Quote:" or "Question:" you should use these to parse the structure of the response, but skip them in your output. Otherwise you should copy each part verbatim. The student should provide a citation at the end of the quote. Make sure you include this citation. 

If you cannot parse the response into the identified parts, return the response exactly as given in `full_response` and `NA` in the other fields. 


==========
Assignment Instructions:

{instructions}

==========
Student Response: 
'
)

## Parse ----
parse_resp = partial(
      generate,
      model = model,
      system = system,
      output = 'text',
      format = schema,
      num_ctx = num_ctx
)

## ~15 sec/response using gemma3:12b
tic()
parsed_resps = map(resps$body, parse_resp, .progress = TRUE)
toc()

parsed_df = parsed_resps |>
      str_squish() %>%
      str_c(collapse = ', ') %>%
      str_c('[', ., ']') |>
      fromJSON() |>
      as_tibble() |>
      mutate(id = 1:n()) |>
      select(id, everything())

## Parsing QA check ----
bind_cols(resps, parsed_df) |>
      rowwise() |>
      mutate(check = {
            map2(body, full_response, adist) |>
                  as.numeric()
      }) |>
      select(student, check, body, full_response) |>
      arrange(desc(check)) |>
      view(title = 'Parsing QA')

# Cluster responses ----
## LLM response schema ----
schema = list(
      type = 'object',
      properties = list(
            think = list(type = 'string'),
            clusters = list(type = 'array', items = list(type = 'string')),
            classified = list(
                  type = 'array',
                  items = list(
                        type = 'object',
                        properties = list(
                              id = list(type = 'integer'),
                              cluster = list(type = 'string')
                        ),
                        required = list('student', 'cluster')
                  )
            )
      ),
      required = list('think', 'clusters', 'classified')
)

## Task prompt ----
system = "The following is a spreadsheet of submissions for a short reading reflection assignment, with these instructions: 

==========
Assignment Instructions:

{instructions}

==========

A previous LLM took the `full_response` and parsed it into the quote, the question, and the answer attempt. Each row starts with a number ID for that response. The columns are separated by a single bar |, and the end of each row is marked by a triple bar |||. 

Your job is to extract thematic clusters from the comments. Focus especially on common questions in the `questions` column. Use the following steps: 

1. First carefully review all the comments. 
2. After reading all the comments, identify 3-5 thematic clusters.
3. Give each cluster a descriptive label of no more than 3 words. 
4. Assign every single comment to exactly one cluster. 
    - Note: You may add an \"unclassified\" cluster if necessary, but try to assign every comment to a thematic cluster if possible. 
5. Conduct two data validation checks: 
    a. Every single comment is classified into a cluster. 
    b. Each comment is classified into only one cluster, and only appears once in the list. 

In your response, use the `think` field to thoroughly document your process for all of the above steps. This field should be detailed, about 200-500 words long. 

Use the `clusters` field to list the labels for the clusters. Then `classified` to do the actual labeling, with the response's ID number, and then the cluster label. I will use the response ID to link your cluster assignment back to the original submission."

prompt = parsed_df |>
      format_delim(delim = '|', eol = '|||')

## Cluster ----
clustered = generate(
      model = model,
      system = system,
      prompt = prompt,
      output = 'text',
      format = schema,
      stream = FALSE,
      num_ctx = num_ctx
)

clustered |>
      fromJSON() |>
      magrittr::extract2('think')

resp_df = fromJSON(clustered)$classified |>
      distinct()

stopifnot(identical(nrow(resp_df), nrow(parsed_df)))

clustered_df = full_join(resp_df, parsed_df, by = 'id') |>
      as_tibble() |>
      arrange(cluster, id)
clustered_df

# Output ----
## 1. build table
tbl = clustered_df |>
      arrange(cluster) |>
      # mutate(response = row_number()) |>
      select(id, quote, question, answer) |>
      tt() |>
      group_tt(i = clustered_df$cluster)
tbl

## 2. write markdown file
outfile = file.path(
      '/Users/danhicks/Google Drive/Teaching/*STE/site/vibe_checks',
      glue('{today()}.md')
)
save_tt(tbl, outfile, overwrite = FALSE)
tbl_md = read_lines(outfile)

## 3. edit markdown file w/ YAML header
## 3a. generate YAML header
header = glue(
      '---
date: {today()}
title: {stamp("Vibe check for Jan 1")(today())}
subtitle: \"Responses by PHIL 006 students, clustered using {model}\"
format: 
  html:
    page-layout: full
---'
) |>
      str_split_1('\n')

## 3b. prepend YAML header
c(header, tbl_md) |>
      write_lines(outfile)
