using LinearAlgebra
export gram_schmidt, least_squares

"""Finds the QR decomposition of X. X should have more rows 
than columns."""
function gram_schmidt(X::Matrix{T}) where T <: Real
  (n, p) = size(X)
  R = zeros(T, p, p)
  Q = copy(X)
  for j = 1:p
    R[j, j] = norm(Q[:, j])
    Q[:, j] = Q[:, j] / R[j, j]
    for k = (j + 1):p
      R[j, k] = dot(Q[:, j], Q[:, k])
      Q[:, k] = Q[:, k] - Q[:, j] * R[j, k]
    end
  end
  return (Q, R)
end

"""Solves the least squares problem of minimizing
||y - X * beta||^2 by the QR decomposition."""
function least_squares(X::Matrix{T}, y::Vector{T}) where T <: Real
  (n, p) = size(X)
  (Q, R) = gram_schmidt(X)
  beta = Q' * y
  for i = p:-1:1 # back substitution
    for j = (i + 1):p
      beta[i] = beta[i] - R[i, j] * beta[j]
    end
    beta[i] = beta[i] / R[i, i]
  end
  return beta
end
