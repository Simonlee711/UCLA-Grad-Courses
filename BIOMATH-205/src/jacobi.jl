using LinearAlgebra, DataStructures
import DataStructures: PriorityQueue, peek
export jacobi_rotation, jacobi

"""Constructs a rotation that zeros out the offdiagonal 
entry b of a 2 x 2 symmetric matrix."""
function jacobi_rotation(a::T, b::T, c::T) where T <: Real
  ratio = (c - a) / (2b)
  if ratio >= zero(T)
    tangent = one(T) / (ratio + sqrt(one(T) + ratio^2))
  else
    tangent = -one(T) / (abs(ratio) + sqrt(one(T) + ratio^2))
  end
  cosine = one(T) / sqrt(one(T) + tangent^2)
  sine = tangent * cosine
  return (cosine, sine)
end

"""Extracts the eigenvalues and corresponding eigenvectors
of the symmetric matrix M. M is destroyed in the process."""
function jacobi(M::Matrix{T}, tol::T) where T <: Real
  n = size(M, 1)
  m = div(n * (n-1), 2)
  key = Vector{Tuple{Int, Int}}(undef, m) # set up keys
  priority = Vector{Float64}(undef, m) # set up priorities
  k = 0
  for i = 1:n
    for j = (i + 1):n
      k = k + 1
      key[k] = (i, j)
      priority[k] = -abs(M[i, j])
    end
  end
  pq = PriorityQueue(zip(key, priority)) # construct queue
  eigenvector = Matrix{T}(I, n, n)
  (i, j), val = peek(pq) # choose biggest off-diagonal entry
  while abs(val) > tol # perform Jacobi rotations until convergence
    (cosine, sine) = jacobi_rotation(M[i, i], M[i, j], M[j, j])
    u = M[i, :]
    v = M[j, :]
    M[i, :] = cosine * u - sine * v
    M[j, :] = sine * u + cosine * v
    u = M[:, i]
    v = M[:, j]
    M[:, i] = cosine * u - sine * v
    M[:, j] = sine * u + cosine * v
    u = eigenvector[:, i]
    v = eigenvector[:, j]
    eigenvector[:, i] = cosine * u - sine * v
    eigenvector[:, j] = sine * u + cosine * v
    for k = 1:n # update priority queue
      if k > i
        pq[(i, k)] = -abs(M[i, k])
      elseif k < i
        pq[(k, i)] = -abs(M[k, i])
      end
      if k > j
        pq[(j, k)] = -abs(M[j, k])
      elseif k < j
        pq[(k, j)] = -abs(M[k, j])
      end
    end
    (i, j), val = peek(pq) # choose next off-diagonal entry
  end
  return (diag(M), eigenvector)
end



