export estimate, predict

"""Estimates the prior and conditional probabilities for
multinomial naive Bayes."""
function estimate(X::Matrix{Int}, class::Vector{Int}, classes::Int,
  pseudo_obs::Float64)
#
  (cases, features) = size(X) # X[i, j] = count of feature j
  freq = zeros(Int, features, classes)
  prior = zeros(classes)
  cond_prob = zeros(features, classes)
  for i = 1:cases
    k = class[i]
    prior[k] = prior[k] + 1.0
    for j = 1:features
      freq[j, k] = freq[j, k] + X[i, j]
    end
  end
  prior = prior / sum(prior)
  for j = 1:classes
    denominator = features * pseudo_obs .+ sum(freq[:, j])
    cond_prob[:, j] = (pseudo_obs .+ freq[:, j]) / denominator
  end
  return (log.(prior), log.(cond_prob))
end

"""Predicts the class of a test case in multinomial naive Bayes."""
function predict(test_case::Vector{Int}, ln_prior::Vector{T}, 
  ln_cond_prob::Matrix{T}) where T <: Real
#
  features = length(test_case)
  ln_posterior = ln_prior
  for i = 1:features
    ln_posterior = ln_posterior + test_case[i] * ln_cond_prob[i, :]
  end
  return argmax(ln_posterior)
end
