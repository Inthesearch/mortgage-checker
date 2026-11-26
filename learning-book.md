# My AI learning book

Started Nov 25, 2026, at my request. **This is my main learning, read on my phone instead of scrolling.** Each chapter is written like an undergraduate textbook section: full explanations, the theory behind the idea, worked examples, the trade-offs, how industry does it, and only a small set of exercises at the end. The laptop sessions are for building; this book is where I learn.

**Rules for chats (Nov 25, 2026):**
- **Every task gets its chapters** (outline below). Chapters for the *next* sub-task are written **before** I start it, so I read on the phone first and build on the laptop second.
- After each session, update that task's chapter with what I actually built and decided ("In my project").
- Same format for every chapter (below). Exercises stay a small part.

**Chapter format** (about 10–20 minutes of reading each):
1. **Why this exists:** the problem it solves.
2. **The core idea:** explained fully, in plain words, with the key terms defined.
3. **How it works:** the mechanism step by step, with the maths where there is any, worked by hand.
4. **Worked example:** a small example done in full.
5. **Design choices and trade-offs:** the options people choose between, and what each costs.
6. **How industry does it:** common tools and practice.
7. **In my project:** where it lives in `mortgage-checker`, what I decided and why.
8. **Common mistakes.**
9. **Key terms:** a short glossary.
10. **Exercises (small):** 3 questions plus one "say it aloud" interview question. Answers at the very end.
11. **Further reading.**

## Outline: every task in the plan
✅ written · ⏳ write before that task starts · 📌 catch-up (already built, chapter still to write)

| Part | Chapter | Task | Status |
|---|---|---|---|
| **A. Foundations (Project 1 steps 0–5)** | A1 Testing LLM apps: why AI needs different tests | Step 0 | 📌 |
| | A2 Rules as code: the mortgage math and pytest | Step 2 | 📌 |
| | A3 Structured output: making an LLM fill a form (Pydantic) | Step 3 | 📌 |
| | A4 Tool calling basics: the LLM asks, Python answers | Step 4 | 📌 |
| | A5 Evaluation suites and prompt fixes | Step 5 | 📌 |
| **B. Evaluation (T1, T2)** | B1 Designing retrieval evals before the build | T1 | 📌 |
| | B2 Code evals, LLM judges and agreement | T2 | 📌 |
| **C. Retrieval (T3)** | 1 Chunking | T3.1 | ✅ |
| | 2 Embeddings | T3.2 | ✅ |
| | 3 Cosine similarity | T3.3 | ✅ |
| | 4 Top-k retrieval and retrieval evals | T3.3–T3.4 | ✅ |
| | 5 The relevance threshold | T3.5 | ✅ |
| | 6 Tools and citations: a second tool the agent can choose | T3.6 | ✅ (read before T3.6) |
| | 7 The refusal path: "the documents don't cover this" | T3.7 | ⏳ |
| **D. Shipping (T4, T6, T7)** | 8 Eval frameworks: Ragas and DeepEval | T4.1 | ⏳ |
| | 9 Where the model runs: hosting, providers and cost | T4.2 | ⏳ |
| | 10 A front end with Streamlit | T4.3 | ⏳ |
| | 11 Deploying and re-evaluating on a new model | T4.4–T4.5 | ⏳ |
| | 12 Containers with Docker | T6 | ⏳ |
| | 13 CI that fails when quality drops | T7 | ⏳ |
| **E. Evaluation depth (T8)** | 14 Building a large eval set | T8 | ⏳ |
| | 15 Adversarial testing: prompt injection and data leaks | T8 | ⏳ |
| | 16 LLM-as-judge done properly | T8 | ⏳ |
| **F. Production (T9, T10)** | 17 Azure OpenAI and cloud deployment | T9 | ⏳ |
| | 18 Observability with Langfuse tracing | T9 | ⏳ |
| | 19 Guardrails | T10 | ⏳ |
| | 20 Model risk in banks: OSFI E-23 and the validation report | T10 | ⏳ |
| **G. Career (T5, T12, T13)** | 21 ISTQB CT-GenAI: the syllabus in one chapter | T5 | ⏳ |
| | 22 Showing your work: resume, LinkedIn, interviews | T12 | ⏳ |
| | 23 Azure AI Foundry and agents (AI-103) | T13 | ⏳ |
| **H. ML basics (T11, T14)** | 24 Classification metrics: precision, recall, F1 | T11 | ⏳ |
| | 25 Training a model with scikit-learn | T11 | ⏳ |
| | 26 PyTorch: tensors and a training loop | T14 | ⏳ |
| **I. Interviews (T15)** | 27 LLM system design | T15 | ⏳ |
| | 28 Agent frameworks: LangGraph and friends | T15 | ⏳ |

---

## Chapter 1 · Chunking: cutting documents into pieces

### 1.1 Why this exists
A language model can only look at a limited amount of text at once (its **context window**), and even when a whole document fits, stuffing it all in makes answers slower, more expensive and less precise: the model has to find the one relevant sentence among thousands. **Retrieval-augmented generation (RAG)** solves this by first *finding* the few passages that matter and then giving only those to the model. But you can't find "the passage that matters" unless the document has already been cut into passages. That cutting is **chunking**, and it is the first design decision in every RAG system. Everything later — embeddings, search, citations, evals — works on chunks, so a bad chunking choice quietly limits everything after it.

### 1.2 The core idea
A **chunk** is a piece of a document small enough to be about one thing, and large enough to make sense on its own. Each chunk is stored with **metadata**: labels that say where it came from. In a RAG system the chunk is the *unit of retrieval*: search returns chunks, the model reads chunks, and answers cite chunks.

Two forces pull in opposite directions:
- **Smaller chunks** are more *precise*: a match means the chunk really is about the question. But a chunk can become so small it loses its meaning ("the amortization period, nor" means nothing alone), and an answer spread across two chunks may be only half found.
- **Larger chunks** carry more *context*: the rule and its exceptions stay together. But one large chunk mixes several topics, so its match score becomes a blurred average, and the model receives extra text that has nothing to do with the question.

Good chunking finds the size where one chunk ≈ one self-contained idea.

### 1.3 How it works
A chunking step has four parts:
1. **Clean the text.** Remove what isn't content: menus, cookie notices, share buttons, footers ("page chrome"). Chrome that survives becomes junk chunks that can win searches by accident.
2. **Choose boundaries.** Decide where one chunk ends and the next begins (section 1.5 lists the options).
3. **Filter.** Drop pieces too short to mean anything (a minimum length), and sometimes split pieces that are too long (a maximum length).
4. **Label.** Give each chunk an id and metadata: which document (`source`), which piece (`section_id`), when the document was fetched (`fetch_date`), and sometimes the page title or heading above it.

Size is usually measured in **characters** or **tokens**. A token is the unit a model reads: roughly ¾ of an English word on average, so 100 tokens ≈ 75 words ≈ 400–500 characters. Context windows and API prices are counted in tokens, which is why many tools measure chunks in tokens too.

### 1.4 Worked example
Take this short (made-up) policy text:

> **Down payment.** The minimum down payment is 5% of the first $500,000 and 10% of any amount above that.
>
> **Insurance.** Homes bought with less than 20% down need mortgage default insurance.

*Option A — one chunk for the whole text.* A question "How much down payment on a $600,000 home?" matches it, but so does "Do I need insurance?". Both get the same blended chunk; the answer has to dig out the right half.

*Option B — split at blank lines.* Two chunks: `policy-001` (down payment) and `policy-002` (insurance). Now each question matches its own chunk strongly. Working the first one: 5% × $500,000 = $25,000; 10% × $100,000 = $10,000; total $35,000 — and the chunk that supports that calculation is exactly `policy-001`, which the answer can cite.

*Option C — split every sentence.* Here it gives the same two chunks, but on a longer page it would separate a rule from its exception ("...unless the home costs over $1.5 million"), and the exception would be lost.

### 1.5 Design choices and trade-offs
| Strategy | How it splits | Good for | Weakness |
|---|---|---|---|
| **Fixed size** | Every N characters or tokens | Simple, predictable size | Cuts mid-sentence; ignores meaning |
| **Fixed size with overlap** | N tokens, each chunk repeating the last few tokens of the previous one | Keeps ideas that cross a boundary | Duplicate text; more chunks to store |
| **Structure-aware** | At headings, paragraphs, list items | Documents with clear structure (regulations, manuals) | Uneven sizes; depends on clean structure |
| **Recursive** | Try paragraphs first; if a piece is still too long, split by sentences, then words | General-purpose default in many libraries | Still blind to meaning |
| **Semantic** | Split where the topic changes (measured with embeddings) | Long flowing text without headings | Slower; harder to predict |

Other choices: the **minimum length** (drop scraps), the **maximum length** (split giants), and **what goes in metadata** (more labels make filtering and citations easier).

### 1.6 How industry does it
- Libraries such as LangChain and LlamaIndex ship ready-made splitters; a recursive splitter with a few hundred tokens per chunk and a small overlap is a common starting point.
- Teams don't trust a default: they **test chunking with retrieval evals** — change the chunk size, re-run the evals, and keep the version that finds the right passage more often. Chunking is a parameter to tune, not a one-time choice.
- In regulated settings, metadata matters as much as text: every answer must point to an exact source and date, so auditors can check it.

### 1.7 In my project
- File: `src/chunker.py`, function `chunk_files(source)`.
- **Cleaning:** `scripts/fetch_sources.py` keeps only the main content tag of each page (`main` for OSFI, `article` for CMHC), set per source in `sources.yaml`. This removed 6 junk chunks from the CMHC page (menus, share widget, feedback form).
- **Boundaries:** structure-aware. The fetch script writes a blank line before every heading (`h1`, `h2`) and list item (`li`); the chunker splits at blank lines. I chose this because both pages are regulations with clear structure, and the CMHC page is one heading followed by a list of rules.
- **Filter:** minimum 30 characters, because "The home is located in Canada." is exactly 30 and is a real rule; 28-character scraps like "the amortization period, nor" are dropped.
- **Labels:** `source`, `section_id` (`osfi_mqr-001` …), `fetch_date` from `sources.yaml`.
- **Result:** 9 OSFI + 9 CMHC = 18 chunks. I predicted 9 and 11; CMHC first gave 15 because of the page chrome.
- **Known weakness:** one OSFI chunk ends with "Watch a video explaining Guideline B-20", a leftover from a link. It doesn't affect any eval, so I left it.

### 1.8 Common mistakes
- Never looking at the chunks: always print them and read a few before embedding.
- Leaving page chrome in: junk chunks can outrank real ones.
- Applying a length rule to each *line* instead of the whole chunk (I did this once: it dropped the date line).
- Changing chunking without re-running the evals, so you can't tell whether it helped.
- Losing the metadata: without `source` and dates you can't cite or check freshness.

### 1.9 Key terms
- **Chunk:** a piece of a document; the unit that search returns.
- **Metadata:** labels stored with a chunk (source, id, date).
- **Context window:** how much text a model can read at once.
- **Token:** the unit a model reads; about ¾ of an English word.
- **Overlap:** text repeated at the start of the next chunk.
- **Page chrome:** non-content parts of a web page (menus, footers, buttons).

### 1.10 Exercises (small)
1. A chunk reads "Date modified: 2026-01-29". Keep it or drop it? Why?
2. You make every chunk 10 times bigger. What happens to the match scores?
3. Your evals got worse after switching to fixed-size chunks of 50 characters. Name the most likely cause.

**Say it aloud (30 seconds):** "How did you chunk your documents, and how would you know if a different chunking was better?"

### 1.11 Further reading
- Chip Huyen, *AI Engineering*, chapter 6 (RAG and agents), the section on chunking (Calgary Public Library, O'Reilly).
- The text-splitter pages in the LangChain or LlamaIndex documentation, to see the strategies above as real code.

*Answers:* 1) Keep it, or better, fold it into the neighbouring chunk: the date is real content (eval case R14 asks for it). It's 25 characters, so a 30-character minimum would drop it if it stood alone — that's why it lives inside `osfi_mqr-009`. 2) Each chunk mixes several topics, so scores become blurred averages: the right chunk stands out less, and the model receives more unrelated text. 3) 50 characters is a few words, so rules are cut in half; the matching piece no longer contains the whole answer.

---

## Chapter 2 · Embeddings: turning text into numbers

### 2.1 Why this exists
Once documents are cut into chunks, we need a way to find the chunk that answers a question. The oldest method is **keyword search**: return chunks that share words with the question. It is fast and still widely used, but it fails whenever people use different words for the same thing. "How much do I need to put down?" and "minimum down payment" share almost no words, yet mean the same. Your eval case R12 failed for exactly this reason. We need a way to compare **meaning**, not spelling. Embeddings are that way.

### 2.2 The core idea
An **embedding** is a list of numbers that represents the meaning of a piece of text. An **embedding model** is a neural network trained to produce these lists so that texts with similar meanings get similar numbers, and unrelated texts get different numbers. The list is called a **vector**, and its length is the model's **dimension**: `nomic-embed-text` produces 768 numbers for any input, whether the input is three words or three paragraphs.

You can picture each vector as a point (or an arrow from the origin) in a space with 768 directions. Texts about down payments sit in one neighbourhood; texts about insurance in another; a recipe for soup far away from both. Searching becomes geometry: embed the question, then find the chunk vectors nearest to it.

No single number in the vector means "money" or "Canada" by itself. Meaning is spread across all 768 numbers together. That is why the numbers look random when you print them, and why you can only compare vectors, not read them.

### 2.3 How it works
1. **Tokenise:** the text is split into tokens (pieces of words).
2. **Encode:** the model, a transformer, reads all the tokens together, so each word's representation depends on its neighbours. "Bank" in "river bank" and "bank account" end up different.
3. **Pool:** the per-token outputs are combined (often averaged) into one vector for the whole text.
4. **Normalise (often):** the vector is scaled to length 1, so only its direction carries meaning. Chapter 3 explains why direction is what we compare.

How does the model learn this? During training it sees huge numbers of text pairs that belong together (a question and its answer, a title and its article) and pairs that don't. It adjusts its internal numbers (its **parameters**, also called weights) so that matching pairs end up close and non-matching pairs far apart. This is called **contrastive learning**. The 768 is not the number of parameters; the model has many millions of parameters, and 768 is just the size of its output.

### 2.4 Worked example
Real embeddings have 768 numbers, but the idea works with 3. Suppose a tiny model gave:
- "minimum down payment" → `[0.9, 0.1, 0.0]`
- "how much must I put down" → `[0.8, 0.2, 0.1]`
- "default insurance premium" → `[0.1, 0.9, 0.2]`

The first two point in almost the same direction even though they share no words except "down"; the third points elsewhere. A search for "how much must I put down" would rank the down-payment chunk first. Chapter 3 computes these similarities by hand.

### 2.5 Design choices and trade-offs
- **Which model:** bigger models usually understand meaning better but are slower and cost more. Local models (like yours, via Ollama) are free and private; hosted models (OpenAI, Azure, Cohere) are often stronger and need no local hardware.
- **Dimension:** more numbers can hold finer distinctions but take more storage and slower search. 384, 768, 1024, 1536 and 3072 are common sizes.
- **Domain fit:** a general model may not know that "GDS" and "gross debt service" are the same. Testing on your own eval cases is the only reliable check.
- **Query vs document roles:** some models (including Nomic's) are trained to embed questions and documents slightly differently, using a short prefix such as `search_query:` or `search_document:` before the text. Using the recommended prefixes can improve retrieval; check the model's card.
- **Re-indexing:** changing the model means re-embedding every chunk. Vectors from different models live in different spaces and must never be compared.

### 2.6 How industry does it
- Chunks are embedded once and stored in a **vector database** (Azure AI Search, pgvector in PostgreSQL, Pinecone, Weaviate), which can search millions of vectors quickly using approximate nearest-neighbour algorithms.
- Many systems combine embeddings with keyword search (**hybrid search**), because keywords still win on exact terms such as product codes, names and numbers like "5.25%".
- Teams compare embedding models on their own retrieval evals before choosing, rather than trusting general leaderboards.

### 2.7 In my project
- File: `src/embed.py`. `embed_text(text)` sends the text to Ollama's `/api/embeddings` endpoint with `nomic-embed-text` and returns 768 numbers. It has a 10-second timeout and turns connection errors, timeouts and HTTP errors into clear messages.
- `build_records(chunks)` adds an `"embedding"` key to each chunk; `save_index` writes all 18 records to `data/rules_index.json`, which is committed to the repo so a fresh clone works without rebuilding.
- At search time the question is embedded with the same model, so question and chunks live in the same space.
- **Not done yet:** I don't use the `search_query:` / `search_document:` prefixes. It is a cheap experiment for later: re-embed, re-run the 15 retrieval evals, and keep it only if the score improves (it might fix R12).

### 2.8 Common mistakes
- Mixing models: building the index with one model and embedding questions with another. Scores become meaningless, with no error.
- Forgetting to rebuild the index after changing the chunks.
- Treating the dimension as quality: a 3072-number vector isn't automatically better than a 768-number one.
- Trusting embeddings for exact facts: "5.25%" and "5.50%" can look very similar to an embedding model.

### 2.9 Key terms
- **Embedding / vector:** a list of numbers that represents meaning.
- **Dimension:** the length of the vector (768 for mine).
- **Embedding model:** the neural network that produces vectors.
- **Parameters (weights):** the learned numbers inside a model; not the same as the dimension.
- **Contrastive learning:** training that pulls matching pairs together and pushes others apart.
- **Vector database:** storage built for fast similarity search.
- **Hybrid search:** combining keyword and embedding search.

### 2.10 Exercises (small)
1. You switch the index to a new embedding model but forget to change the code that embeds questions. What happens?
2. Why can't you look at one number in a vector and say what it means?
3. Which would an embedding find more easily: a chunk that uses a synonym of your question, or a chunk that contains a specific product code you typed?

**Say it aloud (30 seconds):** "What is an embedding, and why use it instead of keyword search?"

### 2.11 Further reading
- Chip Huyen, *AI Engineering*, chapter 6, the section on retrieval and embeddings.
- The `nomic-embed-text` model card on Ollama or Hugging Face (look for the prefix guidance).

*Answers:* 1) Questions and chunks are now in different spaces; scores become meaningless and the wrong chunks are returned, with no error. 2) Meaning is spread across all the numbers together; only comparisons between whole vectors mean anything. 3) The synonym. Exact codes are where keyword search beats embeddings, which is why hybrid search exists.

---

## Chapter 3 · Cosine similarity: how alike two meanings are

### 3.1 Why this exists
Chapter 2 turned every chunk and every question into a vector. Now we need one number that says how close two vectors are, so we can rank chunks. There are several ways to measure closeness; for text embeddings the standard choice is **cosine similarity**, because it compares the *direction* of two vectors and ignores their *length*.

### 3.2 The core idea
Every vector is an arrow from the origin. Two arrows pointing the same way are "about the same thing"; arrows at right angles are unrelated; arrows pointing opposite ways are opposites. Cosine similarity is the cosine of the angle between the two arrows:
- **1.0** — same direction (angle 0°)
- **0** — right angle (90°): unrelated
- **−1.0** — opposite directions (180°)

Real text embeddings rarely go far below 0 and often sit in a narrow band (yours sit roughly between 0.5 and 0.8), so the *ranking* matters more than the raw number.

### 3.3 How it works
Three building blocks:
1. **Dot product:** multiply the vectors position by position and add up. For a = [a₁, a₂] and b = [b₁, b₂]: a·b = a₁b₁ + a₂b₂.
2. **Length (norm):** the length of a is √(a₁² + a₂² + …), Pythagoras extended to any number of dimensions.
3. **Cosine similarity:** cos(a, b) = (a·b) ÷ (length of a × length of b).

Dividing by the lengths is what removes the effect of size: a long arrow and a short arrow in the same direction get 1.0.

**The sign of the dot product** tells you the rough direction: positive means pointing broadly the same way, zero means a right angle, negative means pointing away.

**Shortcut:** if every vector is first normalised to length 1, the denominator is 1 × 1, so cosine similarity *is* the dot product. Vector databases rely on this to search fast.

### 3.4 Worked example
a = [3, 4], b = [4, 3]
- Dot product: 3×4 + 4×3 = 24
- Lengths: √(9+16) = 5 and √(16+9) = 5
- Cosine: 24 ÷ (5×5) = **0.96** — almost the same direction.

a = [3, 0], b = [0, 4]: dot product 0 → cosine **0** (perpendicular).
a = [3, 4], b = [−3, −4]: dot product −9 −16 = −25; lengths 5 and 5 → **−1.0** (opposite).

Why length doesn't matter: [1, 1, 1] against [1, 1, 1] gives 1.0, while [8, 9, 0] against [1, 1, 1] gives 17 ÷ (√145 × √3) ≈ 0.82, even though [8, 9, 0] has much bigger numbers. Direction wins.

### 3.5 Design choices and trade-offs
| Measure | What it compares | When used |
|---|---|---|
| **Cosine similarity** | Angle only | Default for text embeddings |
| **Dot product** | Angle and length | Same as cosine when vectors are normalised; some models are trained for it |
| **Euclidean distance** | Straight-line distance between the points | Common in classic ML; affected by length |

Use the measure the embedding model was trained for; its documentation says which.

**Edge case — the zero vector:** a vector of all zeros has length 0, so the formula divides by zero. Real embeddings are almost never all zeros, but code should decide what to do: return 0, or raise a clear error. Yours currently raises `ZeroDivisionError`, undecided.

### 3.6 How industry does it
- Nobody loops over vectors one by one in production. Libraries such as numpy compute all similarities in one matrix operation, and vector databases use approximate nearest-neighbour indexes (for example HNSW) to avoid comparing against every vector at all.
- Vectors are usually normalised once at indexing time, so search can use the faster dot product.

### 3.7 In my project
- `src/retrieval.py`: `dot_product(a, b)`, `length(a)` and `cosine_similarity(a, b)` in plain Python, written by me on Nov 9; three tests in `tests/test_retrieval.py` (identical → 1.0, `[3,0]`/`[0,4]` → 0.0, `[3,4]`/`[−3,−4]` → −1.0).
- My first test vectors were wrong (I called `[3,4]` and `[4,3]` perpendicular). Working the dot product by hand found it — the function was right, the test was wrong. That is a QA lesson as much as a maths one: work expected values out by hand.
- For speed, `search_rules` uses the numpy version: `(m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q))`, which scores all 18 chunks at once.
- Open: the zero-vector decision and its test.

### 3.8 Common mistakes
- Guessing expected values in tests instead of computing them.
- Comparing raw cosine numbers across different models (each model has its own typical range).
- Forgetting that a high cosine means "similar topic", not "correct answer".

### 3.9 Key terms
- **Dot product:** sum of position-by-position products.
- **Norm / length:** √(sum of squares).
- **Cosine similarity:** dot product divided by the two lengths.
- **Normalise:** scale a vector to length 1.
- **Approximate nearest neighbour (ANN):** fast search that skips most comparisons.

### 3.10 Exercises (small)
1. Compute cos([1, 0], [1, 1]) by hand.
2. Two vectors have a dot product of 0. What does that tell you?
3. All your vectors are normalised. Can you skip the division?

**Say it aloud (30 seconds):** "Why cosine similarity for text, and not just the dot product?"

### 3.11 Further reading
- 3Blue1Brown, "Dot products and duality" (Essence of Linear Algebra, YouTube) for the geometry.
- Chip Huyen, *AI Engineering*, chapter 6 (similarity search).

*Answers:* 1) Dot = 1; lengths 1 and √2; cosine = 1/√2 ≈ 0.71 (45°). 2) They are at a right angle: unrelated. 3) Yes: with length-1 vectors, cosine equals the dot product.

---

## Chapter 4 · Top-k retrieval and retrieval evals: finding and checking

### 4.1 Why this exists
Scoring every chunk is only useful if we then pick the best ones and give them to the model. That's **top-k retrieval**. And because retrieval can silently fail — the right chunk exists but isn't picked — we need **retrieval evals**: test cases that check whether the right chunk was found. Without them, a RAG system can look fine in a demo while missing half the real questions.

### 4.2 The core idea
**Top-k:** score all chunks against the question, sort from highest to lowest, keep the first k. Every chunk is scored whatever k is; k only decides how many come back.

**A retrieval eval case** is a question plus what a correct retrieval must contain: the expected source, and text the right chunk must include. Two kinds:
- **Found cases:** the documents do contain the answer; the right chunk must appear in the top k.
- **Not-covered cases:** the documents don't contain the answer; the system should refuse (chapter 5).

### 4.3 How it works
Common metrics:
- **hit@k:** did any of the top k contain the right chunk? (yes/no per case; report the share of cases)
- **recall@k:** of all the relevant chunks, what share were in the top k? (matters when an answer needs several chunks)
- **precision@k:** of the k chunks returned, what share were relevant?
- **MRR (mean reciprocal rank):** 1 ÷ the rank of the first right chunk, averaged across cases. Rank 1 scores 1, rank 2 scores 0.5, rank 3 scores 0.33. It rewards putting the right chunk *first*, not just somewhere in the top k.

Choosing k is a trade-off: a bigger k raises hit@k (more chances), but hands the model more irrelevant text and can hide a weak search.

**Why save every run:** a history of results lets you see whether a change helped or hurt. Change one thing at a time — chunk size, k, model, threshold — so you know which change caused the effect.

### 4.4 Worked example
Three cases, k = 3. The right chunk's rank in each: case 1 → rank 1, case 2 → rank 3, case 3 → rank 8.
- hit@3: cases 1 and 2 hit, case 3 misses → 2 of 3 = 67%.
- MRR (counting ranks beyond k as 0): (1 + 0.33 + 0) ÷ 3 ≈ 0.44.

Raising k to 10 would make hit@k 3 of 3, but case 3's problem — rank 8 — is still there; you've only hidden it.

### 4.5 Design choices and trade-offs
- **Write the cases first** (before building), so they test the need, not the implementation. Choose the pass bar before seeing results.
- **Diagnose misses before fixing:** is it chunking (answer split across chunks), wording (question and chunk use different words), k (right chunk just below the cut), or the wrong document?
- **Don't tune the test until it passes:** rewording a question to match the chunk makes the score go up and the system no worse — and no better.

### 4.6 How industry does it
- Teams keep a "golden set" of questions with known right chunks, run it on every change, and track the metrics over time, often in CI (your T7).
- Frameworks such as Ragas and DeepEval compute retrieval metrics (context precision and context recall) automatically, sometimes using an LLM to judge relevance (your T4.1 and chapter 8).
- A **reranker** — a second, slower model that re-orders the top results — is a common upgrade when hit@k is fine but the right chunk isn't first.

### 4.7 In my project
- `search_rules(query, k)` in `src/retrieval.py`: loads `data/rules_index.json`, embeds the question, scores all 18 chunks with numpy, returns the top k with `source`, `section_id`, `text` and `score`.
- 15 eval cases (designed in T1, Oct 2026): 11 found, 4 not covered. Run with `evals/run_evals.py`; every run is appended to `evals/retrieval_history.jsonl`, and `evals/show_history.py` prints the last 5 runs with best and worst.
- T3.4 result: found 10 of 11, against my bar of 9.
- **R12 diagnosis (mine):** the right chunk sat at rank 8 with score 0.47 against 0.578 for the top hit, and the question shared only filler words with it. A wording gap, not k and not chunking. I kept it as a known miss rather than rewording the test.
- `evals/score_report.py` prints each case's top score and the rank of its expected chunk.

### 4.8 Common mistakes
- Measuring only hit@k with a large k, which hides poor ranking.
- Rewording the eval question until it passes.
- Changing several things at once, then not knowing what helped.
- Checking that a chunk *contains* some text without noticing the check itself is wrong (my `contains` string-vs-list bug in T3.5 looped over single characters and passed almost anything).

### 4.9 Key terms
- **Top-k:** the k highest-scoring chunks.
- **hit@k, recall@k, precision@k, MRR:** retrieval metrics (4.3).
- **Golden set:** a fixed set of eval questions with known answers.
- **Miss:** a found case whose right chunk isn't in the top k.
- **Reranker:** a second model that re-orders the top results.

### 4.10 Exercises (small)
1. Right chunk at rank 2 in every case. What are hit@3 and MRR?
2. Why might raising k from 3 to 10 make answers worse even if hit@k goes up?
3. A case fails. Name two things to check before changing any code.

**Say it aloud (30 seconds):** "How do you know your retrieval works, and what did you do when a case failed?"

### 4.11 Further reading
- Chip Huyen, *AI Engineering*, chapter 3 (evaluation methodology) and chapter 6 (retrieval).
- The Ragas documentation page on context precision and context recall.

*Answers:* 1) hit@3 = 100%; MRR = 0.5. 2) The model gets more irrelevant text, which can distract it or add wrong details. 3) Any two of: the rank and score of the right chunk; whether the answer is split across chunks; whether the question shares content words with the chunk; whether the case's expected source is right.

---

## Chapter 5 · The relevance threshold: when to say "I don't know"

### 5.1 Why this exists
Top-k retrieval always returns k chunks — even for a question the documents don't cover at all, like "What's the best stock to buy?". If the system passes those chunks to the model, the model will usually produce a fluent, confident, wrong answer. In a mortgage tool that is dangerous. The **relevance threshold** gives the system a way to say "the documents don't cover this" instead.

### 5.2 The core idea
A relevance threshold is one number. If the best chunk's score is below it, the system refuses; at or above it, the system answers from the chunks. This turns a ranking problem into a **yes/no decision**, and every yes/no decision can go wrong in two ways:
- **Wrong answer (false positive):** the threshold is too low, and a not-covered question gets answered.
- **Wrong refusal (false negative):** the threshold is too high, and an answerable question gets refused.

Moving the threshold trades one error for the other. You can't remove both; you choose which costs more.

### 5.3 How it works
1. Run all eval cases and record the top score of each.
2. Look at two groups: found cases (should answer) and not-covered cases (should refuse).
3. If the groups' scores don't overlap, any threshold between them is perfect. Usually they overlap.
4. Pick the threshold where the errors you can't accept are zero, and count the cost of the others.

This is the same idea as a medical test's cut-off or a spam filter's setting. In statistics, a chart of both error rates across all thresholds is called an **ROC curve**, and the chosen point is the **operating point**.

### 5.4 Worked example
Top scores: found cases 0.86, 0.75, 0.71, 0.66, 0.62; not-covered cases 0.68, 0.60, 0.55.
- Threshold 0.70: found answered 3 of 5; not covered refused 3 of 3. No wrong answers, 2 wrong refusals.
- Threshold 0.65: found 4 of 5; not covered refused 2 of 3 (0.68 gets through). One wrong answer.
For a mortgage tool, 0.70 is the safer operating point: a refusal sends someone to the official page; a wrong answer could cost them money.

### 5.5 Design choices and trade-offs
- **Which error is worse** is a business and risk decision, not a technical one. Banks usually choose fewer wrong answers.
- **Thresholds are model-specific:** each embedding model has its own score range, so a new model means re-tuning.
- **Small eval sets make thresholds fragile:** with 15 cases, one case moves the result by about 7 points. More cases (your T8) make it trustworthy.
- **Alternatives and add-ons:** asking the LLM itself whether the chunks answer the question, or a reranker's score, which is often better calibrated than raw cosine.

### 5.6 How industry does it
- Thresholds are tuned on labelled eval sets and re-checked whenever the model or documents change.
- Regulated teams log every refusal and sample them for review, to catch wrong refusals in production.
- Refusals are designed as a feature: a clear message plus a link to the official source.

### 5.7 In my project
- `RELEVANCE_THRESHOLD = 0.689` in `src/retrieval.py`, imported by `evals/run_evals.py` (so T3.7 can import it too).
- My guess before any data was 0.780; at that value only 1 of 11 found cases would pass. The instinct (be strict) was right; the scale was wrong, because these scores sit between about 0.5 and 0.8.
- Result (Nov 21): **12/15** — not covered 4 of 4 ✅, found 8 of 11 (R11 and R14 fail on score; R12 on rank and score). I revised the bars to 4 of 4 and 8 of 11.
- I spotted that found cases never checked the score, so the earlier 14/15 overstated the result; the runner now checks both.
- My decision: wrong answers are worse than refusals, so I accepted R11 and R14 failing instead of lowering the threshold.

### 5.8 Common mistakes
- Lowering the threshold until every found case passes, letting not-covered questions through.
- Tuning on the same few cases you report, then claiming the result generalises.
- Forgetting to re-tune after changing the model, chunking or documents.

### 5.9 Key terms
- **Threshold / cut-off:** the score that separates answer from refuse.
- **False positive / false negative:** wrong answer / wrong refusal (in this setting).
- **Operating point:** the threshold you choose and its error rates.
- **ROC curve:** a chart of both error rates across all thresholds.
- **Calibrated score:** a score whose value reliably means the same thing.

### 5.10 Exercises (small)
1. You raise the threshold from 0.689 to 0.72. What happens to refusals and to found cases?
2. A new embedding model arrives. Can you keep 0.689?
3. Which is worse in your tool, a wrong refusal or a wrong answer? One sentence.

**Say it aloud (30 seconds):** "How did you choose your refusal threshold, and what did it cost you?"

### 5.11 Further reading
- Chip Huyen, *AI Engineering*, chapter 4 (evaluating AI systems), on defining what failure costs.
- Any short introduction to ROC curves and confusion matrices (also chapter 24 of this book, later).

*Answers:* 1) More refusals; some found cases that passed now fail. 2) No: each model has its own score range; re-tune on the evals. 3) A wrong answer: someone could act on a false mortgage rule and lose money, while a refusal sends them to the source.

---

## Chapter 6 · Tools and citations: a second tool the agent can choose (read before T3.6)

### 6.1 Why this exists
Your agent can already call one tool, `assess()`, which does the mortgage maths. Now it also needs to answer rule questions ("What's the minimum qualifying rate?") from the documents. That means a second tool, `search_rules`, and a way for the model to choose between them. And because users must be able to check an answer, every rule answer must show which chunk it came from: a **citation**.

### 6.2 The core idea
**Tool calling** (also called function calling) lets a language model ask the program to run a function. The program describes each tool to the model — its name, what it does, and its inputs, in a structured format (a JSON schema). When a question arrives, the model either answers directly or returns a request such as `search_rules(query="minimum qualifying rate", k=3)`. The program runs the function, sends the result back, and the model writes the final answer using it.

The model never runs code itself. It only *asks*. Your program decides whether to honour the request, which is where validation and safety live.

**Routing** is the model's choice of tool. Maths questions should go to `assess`; rule questions to `search_rules`. The tool's **description** is the main thing the model reads to decide, so writing descriptions is real engineering work.

**Citations** tie each claim to a retrieved chunk id, such as `[osfi_mqr-007]`. A citation is only trustworthy if it refers to a chunk that was actually retrieved in this answer; models sometimes invent plausible-looking ids.

### 6.3 How it works
1. **Describe the tools:** name, description, parameters (types, required or optional).
2. **Send the question plus the tool list** to the model.
3. **Model decides:** answer directly, or return one (or more) tool calls with arguments.
4. **Validate the arguments** (your Pydantic gate already does this for `assess`).
5. **Run the tool** and send the result back as a tool message.
6. **Model writes the answer**, instructed to cite only the chunk ids it was given.
7. **Check the citations in code:** every cited id must be in the set of retrieved ids. If not, flag or reject the answer.

### 6.4 Worked example
Question: "What rate do I have to qualify at if my contract rate is 4%?"
- The model reads both descriptions. `assess` says "computes affordability for a full application"; `search_rules` says "looks up OSFI and CMHC rules". The question is about a rule, so it calls `search_rules(query="qualifying rate", k=3)`.
- Retrieved: `osfi_mqr-004` (score 0.78), `osfi_mqr-005`, `cmhc_requirements-002`.
- Answer: "You must qualify at the higher of your contract rate plus 2 percentage points (6%) or 5.25%, so 6% [osfi_mqr-004]."
- Citation check: `osfi_mqr-004` is in the retrieved set → valid. If the answer had cited `osfi_mqr-012`, which wasn't retrieved, the check fails.

Note the design rule: the arithmetic (4% + 2% = 6%, compared with 5.25%) is exactly the kind of thing tested Python should do. A stricter design has the rule answer cite the chunk and call `assess` for any number.

### 6.5 Design choices and trade-offs
- **Description wording:** vague descriptions cause wrong routing. Say when *not* to use a tool, too.
- **One call or several:** allowing multiple tool calls per question is more capable but harder to test.
- **Citation format:** inline ids are simple and checkable; quoting the exact sentence is stronger but longer.
- **What to do on an invalid citation:** retry, strip the claim, or refuse. For a bank, refusing is the safe default.
- **Small local models** (like llama3.2) route less reliably than large hosted ones, so routing must be evaluated, not assumed.

### 6.6 How industry does it
- All major providers (OpenAI and Azure OpenAI, Anthropic, Google) support tool calling with JSON-schema tool definitions; frameworks such as LangGraph and the OpenAI Agents SDK build multi-step agents on top of it.
- **MCP (Model Context Protocol)** is an emerging standard for exposing tools to any model in the same way.
- Teams evaluate routing with labelled cases ("this question should call X") and check citations automatically — exactly your T4.1 routing check and `citations_valid`.

### 6.7 In my project (planned for T3.6)
- Add `search_rules` to the agent's tool list next to `assess`.
- **CP-T3.6a:** the tool schema and description — I decide the wording, including when to use it versus `assess`.
- **CP-T3.6b:** `citations_valid(answer_ids, retrieved_ids)`, with tests first: all valid, one invented id, no citations.
- Done when a rule question gets a cited answer and the original 20 agent cases still pass.
- *(Update this section after T3.6 with what I actually built and decided.)*

### 6.8 Common mistakes
- Trusting the model's citations without checking them in code.
- Tool descriptions that overlap, so the model can't tell which to use.
- Letting the model do arithmetic that a tested function should do.
- Forgetting to re-run the old cases after adding a tool (new tools can break old routing).

### 6.9 Key terms
- **Tool / function calling:** the model asks the program to run a function.
- **JSON schema:** a structured description of a tool's inputs.
- **Routing:** the model's choice of which tool to call.
- **Citation:** a pointer from a claim to the chunk that supports it.
- **Grounding:** an answer supported by the retrieved text.
- **MCP:** a standard protocol for exposing tools to models.

### 6.10 Exercises (small)
1. The model cites `cmhc_requirements-009`, but only 001, 002 and 004 were retrieved. What should the code do?
2. A maths question gets routed to `search_rules`. What would you change first?
3. Why re-run the 20 original agent cases after adding the new tool?

**Say it aloud (30 seconds):** "How does your agent decide which tool to use, and how do you stop it inventing sources?"

### 6.11 Further reading
- The function-calling (tool use) guide in the OpenAI or Azure OpenAI documentation.
- Chip Huyen, *AI Engineering*, chapter 6 (agents and tools).

*Answers:* 1) Treat the citation as invalid: flag, retry or refuse — never show it as a source. 2) The tool descriptions: make `assess` clearly the tool for calculations and say `search_rules` is for rule lookups only. 3) Adding a tool changes the model's choices, so old questions may now be routed wrongly; the old cases catch that regression.
