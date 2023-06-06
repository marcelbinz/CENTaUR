def distSample(numbers, probabilities, rnd_num):
    # Sampling a single number from a discrete distribution
    #   The possible Numbers in the distribution with their resective
    #   Probabilities. rndNum is a randomly drawn probability
    #
    #   Conditions on Input (not checked):
    #   1. Numbers and Probabilites correspond one to one (i.e. first number is
    #   drawn w.p. first probability etc)
    #   2. rndNum is a number between zero and one
    #   3. Probabilites is a probability vector

    cum_prob = 0
    sampled_int = 0
    while rnd_num > cum_prob:
        cum_prob += probabilities[sampled_int]
        sampled_int += 1
    return numbers[sampled_int - 1]
