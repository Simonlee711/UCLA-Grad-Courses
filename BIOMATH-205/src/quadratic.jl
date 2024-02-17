export quadratic

""" Computes the roots of the quadratic a*x^2+b*x+c when |a|>0."""
function quadratic(a::T, b::T, c::T) where T <: Real
  d = b^2 - 4a * c
  if d > zero(T)
    if b >= zero(T)
      r1 = (-b - sqrt(d)) / (2a)
    else
      r1 = (-b + sqrt(d)) / (2a)
    end
    r2 = c / (r1 * a)
    return (r1, r2)
  else
    return (-b + sqrt(d + 0im)) / (2a), (-b - sqrt(d + 0im)) / (2a)
  end
end
