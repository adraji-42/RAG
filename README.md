# RAG

## Foreword
A language model is frozen in time. Everything it “knows” was fixed the day its training
ended. Ask it about a library released last month, a private codebase, or yesterday’s
incident report, and it will either shrug or invent a confident answer out of thin air.
Retraining a model every time the world changes is absurdly expensive, so we cheat.
Instead of stuffing more knowledge into the model, we let it reach out for the right
information at the moment it answers, the same way you don’t memorise a whole manual
but you know which page to open.
That sounds easy until you try it. The “manual” here is an entire codebase: thousands
of files, hundreds of thousands of lines. Finding the two or three snippets that actually
answer a question, and only those, is a needle-in-a-haystack problem. Search too little
and you miss the answer; search too much and you drown the model in noise.
Our intuitions are bad at this kind of scale, but good engineering is not. Retrieving the
right evidence from a mountain of data, then making a small model answer faithfully
from it, is what this project is about.

## Description
Retrieval-Augmented Generation (RAG) is an NLP framework that combines the strengths of information retrieval with large language models (LLMs) to enhance the accuracy and factual grounding of generated text.

## Resources
[RAG](https://www.geeksforgeeks.org/nlp/what-is-retrieval-augmented-generation-rag/)
[Chunking](https://medium.com/@dev_tips/25-chunking-tricks-for-rag-that-devs-actually-use-12bebd0375bc)
[Incremental Indexing](https://medium.com/@vasanthank29/incremental-indexing-strategies-for-large-rag-systems-e3e5a9e2ced7)
[Fire Python](https://www.geeksforgeeks.org/python/python-fire-module/)
