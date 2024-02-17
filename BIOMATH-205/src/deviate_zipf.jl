export zipf_deviate

"""Generates n random deviates from the Zipf(s) distribution."""
function zipf_deviate(s::T, n::Int) where T <: Real
  x = zeros(Int, n);
  r = one(T) / (one(T) - s)
  c = s / (s - one(T))
  y = zero(T)
  for i = 1:n
    for trial = 1:1000
      y = c * rand(T)
      if y > one(T)
        y = (y * (one(T) - s) + s)^r # quantile value
      else
        x[i] = 1
        break
      end
      if rand(T) < (y / ceil(Int, y))^s # rejection test
        x[i] = ceil(Int, y) # conversion to discrete variate
        break
      end
    end
  end
  return x
end
