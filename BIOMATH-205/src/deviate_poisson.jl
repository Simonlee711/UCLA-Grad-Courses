export poisson_deviate

"""Generates Poisson random deviates with mean mu."""
function poisson_deviate(mu::T, n::Int) where T <: Real
  x = zeros(Int, n)
  x[1] = round(Int, mu)
  for i = 1:(n - 1)
    u = rand(T, 2)
    if u[1] < one(T) / 2
      if u[2] < x[i] / mu
        x[i + 1] = x[i] - 1
      else
        x[i + 1] = x[i]
      end 
    else
      if u[2] < mu / (x[i] + 1) 
        x[i + 1] = x[i] + 1
      else
        x[i + 1] = x[i]
      end
    end
  end
  return x
end
