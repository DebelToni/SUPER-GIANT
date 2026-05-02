# Ivancho Brain

Experimental graph recurrent language model for TinyStories.

Each token is injected into `sqrt(num_heads)` graph heads. The graph then runs fixed inner recurrent steps. Every step produces output-head logits and a soft halting weight from a confidence threshold. The training loss uses all inner-step logits, weighted by the halting signal, plus a small expected-step penalty.

The sparse graph is fixed after initialization: every head has at least one outgoing edge, every head receives at least one incoming edge, and the output head is a normal graph node with incoming and outgoing edges.
