# Tsetlin Machine

The paper [The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic](https://arxiv.org/abs/1804.01508) really piqued my interest, despite some weird mistakes and unclear statements. To learn more about it, I decided I wanted to implement it. I spent a few hours so far and I attempted to replicate the noisy XOR problem and while I got close I could not replicate it directly (I constantly get to around 95% accuracy instead of 99.3%). I assume I made a small mistake somewhere but I don't have time the following days to hunt for the bug.

Despite this, I know others are interested in this paper and the concept is really intriguing. This is why I decided to publish this code anyway. I focused on readability over performance, where there is a lot to gain. I attempted to start MNIST but currently it would take about 285 days to do 300 epochs on a single bit resolution input.

The experiments.py file has an example how to use it and I attempted to document wherever it might not be clear what is happening.

The hyperparameters you can pass along together with the names in the paper are:

- `number_clauses`: The number of Automata Teams also known as the units that learn individual propositions.
- `number_action_states`: The amount of states a single Automata can take, smaller state spaces are more flexible and adaptive but also more noisy.
- `precision`: Influences how often to reward or penalize automata.
- `threshold`: If the summation decision value is this far away in the correct decision, stop updating