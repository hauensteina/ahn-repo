# A Transformer model to double the letter A in the input

ABCAB -> AABCAAB

Training samples:

{ABCAB,AABCAAB}{AXA,AAXAA}{...

Each batch  
Alphabet: [A-T,|{}]

(1) ABCAB,AABCAAB\nAKA,AAKAA         # Input file with training data
(2) {ABCAB,AABCAAB}{AKA,AAKAA}{CD    # Batch generation (may be cut off, len = BLOCK_SZ)
(3) x =  {ABCAB,AABCAAB}{AKA,AAKAA}{CD
(4) y0 = ABCAB,AABCAAB}{AKA,AAKAA}{CDE
(5) y1 = 000000AABCAAB}{0000AAKAA}{000 

Not sure (5) is necessary or saves any time. Try it out.
I think there will be random high losses if I skip (5).

A 0 stands for pass-through, i.e. if the model predicts a zero, we replace the zero with the input character. In/Out pairs are enclosed in curlies. The comma triggers beginning of output. A closing curly predicts an opening curly.

Generation looks like
```
    {AB, -> 0000A -> {AB,A
    {AB,A -> 0000AA -> {AB,AA
    {AB,AA -> 0000AAB -> {AB,AAB
    {AB,AAB -> 0000AAB} -> {AB,AAB}
```

We terminate on a }.
The result can be found between , and }.



