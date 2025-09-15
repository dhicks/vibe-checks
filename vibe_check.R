library(tidyverse)
library(rcanvas)

library(ollamar)
library(jsonlite)
library(glue)
library(cluster)
library(tictoc)

library(tinytable)

# LLM parameters ----
embed_model = 'snowflake-arctic-embed2' ## for generating embeddings
inference_model = 'gemma3:12b' ## for labeling clusters and parsing responses
num_ctx = 16000 ## context window

# Retrieve responses ----
## Canvas parameters ----
## STE Fall 2025
## <https://catcourses.ucmerced.edu/courses/35976>
course_id = '35976'
## Vibe check for Week 04
assignment_id = '503500'

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
      ## Prefix each name w/ the number they get in Speed Grader
      mutate(
            idx = {
                  row_number() %>%
                        str_pad(., pad = '0', width = max(str_length(.)))
            },
            prefix = glue('{idx}-{sortable_name}')
      )

## Get submissions ----
resps = get_submissions(course_id, 'assignments', assignment_id) |>
      as_tibble() |>
      filter(!grade == 'incomplete') |>
      select(id = user_id, body) |>
      filter(!is.na(body)) |>
      ## Canvas HTML garbage
      mutate(body = {
            body |> xfun::strip_html() |> textutils::HTMLdecode()
      }) |>
      left_join(students_df, by = 'id') |>
      select(student = prefix, body) |>
      mutate(id = row_number())

## Claude-generated responses for development
# resps = 'gen_responses.txt' |>
#       read_file() |>
#       str_split_1('==========')

# Cluster responses ----
## Construct embeddings ----
tic()
embeddings = embed(
      embed_model,
      resps$body,
      num_ctx = num_ctx,
      truncate = FALSE
)
toc()

sim_mx = crossprod(embeddings)
dist_mx = as.dist(1 - sim_mx)

## Hierarchical clustering ----
clust = hclust(dist_mx)
plot(clust)
rect.hclust(clust, k = 6)

## Maximize silhouette score
## Tends to create larger clusters than I want?
# silhouette_scores <- sapply(2:10, function(k) {
#       clusters <- cutree(clust, k = k)
#       mean(silhouette(clusters, dist_mx)[, 3])
# })
# which.max(silhouette_scores) + 1

## Identify the lowest value of k st the largest cluster is <= 5
max_freq <- function(x) {
      max(table(x))
}
first_thresh = function(x, thresh = 5) {
      which(x <= thresh) |>
            first()
}

k = map_int(
      1:10,
      ~ {
            cutree(clust, k = .x) |>
                  max_freq()
      }
) |>
      first_thresh(6)

message(glue('k = {k}'))

plot(clust)
rect.hclust(clust, k = k)

clusters_df = resps |>
      mutate(cluster_idx = cutree(clust, k = k)) |>
      relocate(cluster_idx, id) |>
      arrange(desc(cluster_idx))

# Labels ----
## Assignment instructions ----
instructions = '/Users/danhicks/Google Drive/Teaching/*STE/site/assignments/vibe_check.md' |>
      read_file()

## LLM response schema ----
label_schm = list(
      type = 'object',
      properties = list(
            think = list(type = 'string'),
            clusters = list(
                  type = 'array',
                  items = list(
                        type = 'object',
                        properties = list(
                              cluster_idx = list(type = 'integer'),
                              cluster_label = list(type = 'string')
                        ),
                        required = list('cluster_idx', 'cluster_label')
                  )
            )
      ),
      required = list('think', 'clusters')
)

## Labeling task prompt
label_sys = glue(
      "The following is a spreadsheet of submissions for a short reading reflection assignment, with these instructions: 

==========
Assignment Instructions:

{instructions}

==========
      
Hierarchical clustering has been applied, assigning each submission to a cluster. Each row starts with a numeric cluster index, the student's name, and the body of their response. The columns are separated by a single bar |, and the end of each row is marked by a triple bar |||. 

Your task is to come up with labels for each cluster. Use the `think` field to document your work. This field should be detailed, 500-1000 words long. 

First note to yourself the total number of clusters. Each cluster must be assigned exactly one label. 

Explicitly consider the contents of each submission in the cluster. Brainstorming 2-3 potential labels for choosing one. A good label will be no more than 5 words long, and help the students understand both the major topic or theme of the cluster and also how it's distinctive from other clusters. 

The clusters, their labels, and the submission text will be used to create a quick-reference table for discussion in class. 

==========
"
)

label_prompt = clusters_df |>
      format_delim(delim = '|', eol = '|||')

## Generate labels ----
tic()
labels_resp = generate(
      model = inference_model,
      system = label_sys,
      prompt = label_prompt,
      output = 'text',
      format = label_schm,
      stream = TRUE,
      num_ctx = num_ctx,
      num_predict = num_ctx
      # seed = 2025091200
)
toc()

labels_df = fromJSON(labels_resp)$clusters

## Labels QA
left_join(clusters_df, labels_df, by = 'cluster_idx') |>
      as_tibble() |>
      relocate(cluster_label, .before = student) |>
      relocate(student, .before = cluster_idx) |>
      view('Labels QA')

stop('Confirm labels before moving on')

# Parse responses ----
## LLM response schema ----
parse_schm = list(
      type = 'object',
      properties = list(
            full_response = list(type = 'string'),
            quote = list(type = 'string'),
            question = list(type = 'string'),
            answer = list(type = 'string')
      ),
      required = list('full_response', 'quote', 'question', 'answer')
)

## Task prompt ----
parse_sys = glue(
      'A college student was given a short, structured reading-reflection assignment. Your job is to identify the parts of the student\'s response, and arrange them into a JSON structure for further processing. 

Pay attention to the structure of the response, not the content. Do not address the questions or provide any commentary on the quote or their answer. 

The `full_response` field should contain the response exactly as given. For the other fields, if the response includes section numbers or paragraph headers like "Quote:" or "Question:" you should use these to parse the structure of the response, but skip them in your output. Otherwise you should copy each part verbatim. The student should provide a citation at the end of the quote. Make sure you include this citation. 

If you cannot parse the response into the identified parts, return the response exactly as given in `full_response` and `NA` in the other fields. 


==========
Assignment Instructions:

{instructions}

==========
'
)

## Parse ----
parse_resp = partial(
      generate,
      model = inference_model,
      system = parse_sys,
      output = 'text',
      format = parse_schm,
      num_ctx = num_ctx,
      seed = 20250911
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
      mutate(id = row_number()) |>
      select(id, everything())

## Parsing QA check ----
full_join(resps, parsed_df, by = 'id') |>
      rowwise() |>
      mutate(check = {
            ## Levenshtein distance between original and "full response" in model output
            map2(body, full_response, adist) |>
                  as.numeric()
      }) |>
      select(student, check, body, full_response, id) |>
      arrange(desc(check)) |>
      view(title = 'Parsing QA')

stop('Confirm parsing before moving on')

# Combine labeled clusters + parsed responses ----
comb_df = labels_df |>
      as_tibble() |>
      right_join(clusters_df, by = 'cluster_idx') |>
      replace_na(list(cluster_label = 'Other Questions')) |>
      mutate(cluster_label = {
            cluster_label |>
                  fct_infreq() |>
                  fct_rev()
      }) |>
      select(cluster = cluster_label, id, student, body) |>
      full_join(parsed_df, by = 'id') |>
      select(!full_response) |>
      arrange(cluster)

comb_df


# Output ----
## Markdown table for website ----
## 1. build table
tbl = comb_df |>
      select(id, quote, question, answer) |>
      ## Handle paragraph breaks w/in submission parts
      mutate(across(everything(), ~ str_replace_all(.x, '\n', '</br>'))) |>
      tt() |>
      group_tt(i = as.character(comb_df$cluster))
tbl
stop('check for paragraphs')

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
subtitle: \"Responses by PHIL 006 students, clustered using {embed_model} and {inference_model}\"
format: 
  html:
    page-layout: full
---'
) |>
      str_split_1('\n')

## 3b. prepend YAML header
c(header, tbl_md) |>
      write_lines(outfile)

## CSV with names and clustered, parsed questions ----
csv_out = file.path('csv', glue('{today()}.csv'))

write_excel_csv(comb_df, csv_out)
