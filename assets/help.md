

## What is this?

This is a search engine that lets you ask deeper questions of Fora conversations than you
ordinarily would with keyword search.   Your search query can be in natural language and
can be concrete (such as "jobs involving animals") or more abstract
(such as "themes of gentrification").  In addition to to a list of relevant speaker turns,
an "analysis" (chatbot output with citations) will be produced that ties the
speaker turns together to answer your question.  Try some of the canned examples to start.

## What can't you do?

Queries involving mood, intent, affect, or other aspects that are implied by but
not expressed directly in the text will not work well;
for example, "examples of people sounding jealous" will mostly pull up people talking about
jealousy or saying that they're jealous, but not necessarily people who are jealous.


## How does it work?

This tool uses concepts from the emerging area of
retrieval-augmented generation.  A query goes through four stages:

- **Retrieval**:   First, a set of speaker turns are retrieved using a conventional embedding-based nearest-neighbor search.  The embeddings for the ~350k Fora speaker turns are precomputed using a local LM and stored in an efficient vector database.   The embedding for the query is computed at runtime and matched against this db.

- **Selection**:   The results of retrieval are listed on the left, sorted by distance to the query embedding.   The top N=100 speaker turns are selected to go on to the "analysis" phase.    (In a future version it could take advantage of the structure of the conversations:  for example, it could pull in the other​ speaker turns from the speakers who had turns that matched, so that background information about the speakers is made available to the analysis below.   Or it could pull in speaker turns before and after the matches to give the analysis more context.   But I haven't implemented these ideas yet.)

- **LLM generation** (the "analysis"):   The selected speaker turns from (2) are put into a prompt for an LLM, along with the original user query (and some instructional text that ties the two together.).  The prompt instructs the LLM to cite its sources by the ID of the speaker turn.  We can then take the completion provided by the LLM and inject footnotes that link back to the retrieval results on the left.

- **Re-sorting / layout**:  Given the results of the analysis, we can be more creative about how we lay out the original retrieval results.  For now we go back and sort the results by the number of citations received in the analysis;  this adds a kind of salience factor to the overall ranking function that is not captured by the embedding-based relevance ranking alone, which is especially useful for more abstract queries (e.g. "What are some themes related to gentrification").   There are many directions we could go in this phase.   In theory we could use the citation info, and the known structure of the conversations, to lay the results out more visually than just a linear list.  
