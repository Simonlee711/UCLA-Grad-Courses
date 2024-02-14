using LinearAlgebra, ForwardDiff, Printf, SpecialFunctions

"""
Computes the values of a function f over a user defined grid of 
points or the constrained optimum of f by the method of recursive 
quadratic programming. The Hessian d2f of f is approximated by Davidon's 
secant update unless an exact value is supplied by the user. Users 
must supply the objective function fun. Parameter initial values and 
constraints are passed through the parameter data structure.
"""
function search(fun::Function, initial, io)
  #
  # Set variables controlling convergence and numerical differentiation.
  #
  conv_crit = 1e-6
  conv_tests = 4
  max_iters = 1000
  max_steps = 3
  max_step_length = Inf
  dp = 1e-6
  small = 1e-5
  #
  # Write a logo.
  #
   println(io,"\n                       Search, Julia Version\n")
   println(io,"                 (c) Copyright Kenneth Lange, 2022\n")
  #
  # Initialize the current problem.
  #
  (grid, pname, par, pmin, pmax, constraint, level,
     travel, goal, title, cases, standard_errors) = initial()
  travel = lowercase(travel)
  goal = lowercase(goal)
  (constraints, pars) = size(constraint)
  points = size(grid, 1)
  #
  # Write a header.
  #
  println(io,"\nTitle: ",title)
  println(io,"Grid or search option: " * travel)
  println(io, "Minimize or maximize: " * goal)
  #
  # Check for bound and constraint errors.
  #
  error = check_constraints(constraint, level, pname, par, pmin,
    pmax, io, travel)
  if error; return par; end
  #
  # Check for other errors.
  #
  if goal != "minimize" && goal != "maximize"
    throw(ArgumentError(
    "The variable goal ($goal) must be minimize or maximize.\n \n"))
  end
  if travel != "grid" && travel != "search"
    throw(ArgumentError(
    "The variable travel ($travel) must be grid or search.\n \n"))
  end
  if constraints < 0
    throw(ArgumentError(
    "The number of constraints ($constraints) must be nonnegative.\n \n"))
  end
  if pars < 0
    throw(ArgumentError(
      "The number of parameters ($pars) must be nonnegative.\n \n"))
  end
  if points < 0
    throw(ArgumentError(
      "The number of grid points ($points) must be nonnegative.\n \n"))
  end
  #
  # Create variables required in optimization.
  #
  f = 0.0
  df = zeros(pars)
  d2f = Matrix{Float64}(I, pars, pars)
  df_old = zeros(pars)
  par_old = zeros(pars)
  if travel == "grid"
    function_value = zeros(points)
  else
    function_value = zeros(2)
  end
  #
  # Compute function values over a user defined grid of points.
  # Keep track of the optimal point and optimal function value.
  #
  fmin = Inf
  iter_min = 0
  best_point = zeros(pars)
  if travel == "grid"
    for iteration = 1:points
      par = vec(grid[iteration, :])
      f = fun(par)
      iteration_output!(io, f, iteration, par, pname, 0)
      function_value[iteration] = f
      if goal == "maximize"
        f = - f
      end
      if f < fmin
        fmin = f
        iter_min = iteration
        copyto!(best_point, par)
      end
    end
    #
    # Return the optimal point and optimal function value.
    #
    if goal == "minimize"
      println(io, " \nThe minimum function value of ", 
        round(fmin; digits = 5)," occurs at iteration ",iter_min, ".")
      return (best_point, fmin)
    else
      println(io, " \nThe maximum function value of ", 
        round(-fmin; digits = 5)," occurs at iteration ",iter_min, ".")
      return (best_point, - fmin)
    end
    #
    # Otherwise minimize the objective function. df and d2f
    # are the first two differentials. ker is the kernel of
    # the constraint matrix.
    #
  else
    (matrix_u, d, matrix_v) = svd(constraint, full = true)
    ker = matrix_v[:, constraints + 1:pars]
    conv_count = 0
    #
    # Fetch the first function value and the differential.
    # Output the first iteration.
    #
    iteration = 1
    f = fun(par)
    grad = ForwardDiff.gradient(fun, par)
    iteration_output!(io, f, iteration, par, pname, 0)
    function_value[1] = f
    #
    # Copy the user supplied first and second differentials.
    #
    if goal == "minimize"
      copyto!(df, grad)
    else
      f = - f
      copyto!(df, - grad)
    end
    #
    # Preserve the current point.
    #
    copyto!(best_point, par)
    fmin = f
    iter_min = 1
    #
    # Enter the main iteration loop.
    #
    for iteration = 2:max_iters
    #
    # Solve the quadratic programming problem. If cycles = 0,
    # then the quadratic programming algorithm has failed. Reset
    # the Hessian approximation to the identity and try again.
    #
      for j = 1:2
        tableau = create_tableau(d2f, df, constraint, level, par)
        (cycles, delta) =
        quadratic_program(tableau, par, pmin, pmax, pars, constraints)
        #
        # Check whether quadratic programming succeeded.
        #
        if cycles > 0
          break
        else
          c = Inf
          for i = 1:pars
            if d2f[i, i] > 0.0; c = min(c, d2f[i, i]); end
          end
          #
          # In an emergency revert to steepest descent.
          #
          if c >= Inf
            d2f = Matrix{Float64}(I, pars, pars)
          #
          # Otherwise, bump up the diagonal entries of the Hessian.
          #
          else
            for i = 1:pars
              if d2f[i, i] <= c; d2f[i, i] = c * (1.0 + rand()); end
            end
          end
        end
      end
      #
      # Compute a new function value. If it represents a sufficient
      # decrease, then accept the new point. Otherwise backtrack by
      # fitting a quadratic through the two current points with the
      # given slope at the first point. Quit, if necessary, after the
      # maximum number of steps.
      #
      l2_norm = norm(delta)
      if l2_norm > max_step_length
        delta = max_step_length * delta / l2_norm
      end
      d = min(dot(df, delta), 0.0)
      copyto!(df_old, df)
      copyto!(par_old, par)
      f_old = f
      t = 1.0
      steps = -1
      for step = 0:max_steps
        steps = steps + 1
        #
        # Fetch another function value, and if provided, the user
        # supplied second differential.
        #
        par = par_old + t * delta
        f = fun(par)
        grad = ForwardDiff.gradient(fun, par)
        if goal == "minimize"
          copyto!(df, grad)
        else
          f = - f
          copyto!(df, - grad)
        end
        #
        # Test for a sufficient decrease in the objective.
        #
        if f <= f_old + t * d / 10.0 || step == max_steps
          break
        else
          t = max( -d * t * t / (2.0 * (f - f_old - t * d)), t / 10.0)
        end
      end
      #
      # Output the current iteration, and update the best point.
      #
      if goal == "minimize"
        iteration_output!(io, f, iteration, par, pname, steps)
      else
        iteration_output!(io, - f, iteration, par, pname, steps)
      end
      if f < fmin
        fmin = f
        iter_min = iteration      
        copyto!(best_point, par)
      end
      #
      # Check the convergence criterion. If it has been satisfied a
      # sufficient number of times, then exit the main loop. Otherwise,
      # output the current iteration and continue the search.
      #
      if abs(f_old - f) > conv_crit
        conv_count = 0
      else
        conv_count = conv_count + 1
      end
      if conv_count >= conv_tests
        break
      end
      #
      # Update the Hessian by Davidon's secant method.
      #
      delta = par - par_old
      u = df - df_old - d2f * delta
      c = dot(u, delta)
      if c^2 > (1e-8) * dot(u, u) * dot(delta, delta)
        c = -1.0 / c
        #
        # Ensure that the updated Hessian is positive definite on
        # the kernel of the constraint matrix.
        #
        matrix = inv(ker' * d2f * ker)
        v = ker' * u
        d = dot(v, matrix * v)
        c = min(c, 0.8 / d)
        d2f = d2f - c * u * u'
      end
    end
  end
  #
  # Output the optimal function value.
  #
  if goal == "minimize"
    println(io, " \nThe minimum function value of ", 
      round(fmin; digits = 5)," occurs at iteration ", iter_min, ".")
    function_value[2] = fmin
  else
    println(io, " \nThe maximum function value of ", 
      round(- fmin; digits = 5)," occurs at iteration ", iter_min, ".")
    function_value[2] = - fmin
  end
  #
  # If the asymptotic covariance matrix is desired, then
  # record which parameters occur on a boundary
  #
  boundary = falses(pars)
  if standard_errors
    for i = 1:pars
      boundary[i] = (par[i] <= pmin[i] + small || par[i] >= pmax[i] - small)
    end
    boundaries = sum(boundary)
    #
    # Approximate the columns of the second differential by 
    # forward differences of the gradient.
    #
    for i = 1:pars
      par[i] = par[i] + dp
      grad = ForwardDiff.gradient(fun, par)
      d2f[:, i] .= (grad - df) / dp
      par[i] = par[i] - dp
    end
    if goal == "maximize"
      d2f .= -d2f
    end
    #
    # Adjust the Hessian for a least squares problem by dividing
    # by the residual mean square.
    #
    reduced = pars - constraints - boundaries
    if cases > 0
      n = cases - reduced
      if n > 0 && f > 0.0
        sigma_sq = 2.0 * f / n
        d2f = d2f / sigma_sq
      else
        fill!(d2f, 0.0)
      end
    end
    #
    # After these preliminaries, compute the asymptotic covariances.
    #
    if reduced > 0
      asy_cov = asymptotic_covariances(io, constraint, boundary, d2f, pname)
    else
      write(io, " \n The asymptotic covariance matrix is undefined.")
      println(io, "")
    end
  end
  #
  # Return the optimal point and optimal function value.
  #
  if goal == "minimize"
    return (best_point, fmin)
  else
    return (best_point, - fmin)
  end
end # function mendel_search

"""
Calculates the tableau used in minimizing the quadratic 
0.5 x' Q x + r' x, subject to Ax = b and parameter lower 
and upper bounds.
"""
function create_tableau(matrix_q::Matrix{Float64}, r::Vector{Float64},
  matrix_a::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64})

  m = size(matrix_a, 1)
  #
  # Create the tableau in the absence of constraints.
  #
  if m == 0
    tableau = [matrix_q (-r); -r' 0]
  else
  #
  # In the presence of constraints compute a constant mu via
  # the Gerschgorin circle theorem so that Q + mu * A' * A
  # is positive definite.
  #
    (matrix_u, d, matrix_v) = svd(matrix_a, full = true)
    matrix_p = matrix_v * matrix_q * matrix_v'
    mu = 0.0
    for i = 1:m
      mu = max((norm(matrix_p[:, i], 1) - 2.0 * matrix_p[i, i]) / d[i]^2, mu)
    end
    mu = 2.0 * mu
    #
    # Now create the tableau.
    #
    tableau = [matrix_q + mu * matrix_a' * matrix_a matrix_a' (-r);
               matrix_a zeros(m, m) b - matrix_a * x;
               -r' (b - matrix_a * x)' 0]
  end
  return tableau
end # function create_tableau

"""
Solves the p-dimensional quadratic programming problem
 min [df * delta + 0.5 * delta' * d^2 f * delta]
 subject to: constraint * delta = 0 and pmin <= par + delta <= pmax.
See: Jennrich JI, Sampson PF (1978) "Some problems faced in making
a variance component algorithm into a general mixed model program."
Proceedings of the Eleventh Annual Symposium on the Interface.
Gallant AR, Gerig TM, editors. Institute of Statistics,
North Carolina State University.
"""
function quadratic_program(tableau::Matrix{Float64}, par::Vector{Float64},
  pmin::Vector{Float64}, pmax::Vector{Float64}, p::Int, c::Int)

  delta = zeros(Float64, size(par))
  #
  # See function create_tableau for the construction of the tableau.
  # For checking tolerance, set diag to the diagonal elements of tableau.
  # Begin by sweeping on those diagonal elements of tableau corresponding
  # to the parameters. Then sweep on the diagonal elements corresponding
  # to the constraints. If any parameter fails the tolerance test, then
  # return and reset the approximate Hessian.
  #
  small = 1e-5
  tol = 1e-8
  d = diag(tableau)
  for i = 1:p
    if d[i] <= 0.0 || tableau[i, i] < d[i] * tol
      return (0, delta)
    else
      sweep!(tableau, i, false)
    end
  end
  swept = trues(p)
  for i = p + 1:p + c
    if tableau[i, i] >= 0.0
      return (0, delta)
    else
      sweep!(tableau, i, false)
    end
  end
  #
  # Take a step in the direction tableau(i, end) for the parameters i
  # that are currently swept. If a boundary is encountered, determine
  # the maximal fractional step possible.
  #
  cycle_main_loop = false
  for iteration = 1:1000
    a = 1.0
    for i = 1:p
      if swept[i]
        ui = tableau[i, end]
        if ui > 0.0
          ai = pmax[i] - par[i] - delta[i]
        else
          ai = pmin[i] - par[i] - delta[i]
        end
        if abs(ui) > 1e-10
          a = min(a, ai / ui)
        end
      end
    end
    #
    # Take the fractional step for the currently swept parameters, and
    # reset the transformed partial derivatives for these parameters.
    #
    for i = 1:p
      if swept[i]
        ui = tableau[i, end]
        delta[i] = delta[i] + a * ui
        tableau[i, end] = (1.0 - a) * ui
        tableau[end, i] = tableau[i, end]
      end
    end
    #
    # Find a swept parameter that is critical, and inverse sweep it.
    # Go back and try to take another step or fractional step.
    #
    cycle_main_loop = false
    for i = 1:p
      critical = pmin[i] >= par[i] + delta[i] - small
      critical = critical || pmax[i]<= par[i] + delta[i] + small
      if swept[i] && abs(tableau[i, i])>1e-10 && critical
        sweep!(tableau, i, true)
        swept[i] = false
        cycle_main_loop = true
        break
      end
    end
    if cycle_main_loop; continue; end
    #
    # Find an unswept parameter that violates the KKT condition
    # and sweep it. Go back and try to take a step or fractional step.
    # If no such parameter exists, then the problem is solved.
    #
    for i = 1:p
      ui = tableau[i, end]
      violation = ui > 0.0 && pmin[i] >= par[i] + delta[i] - small
      violation = violation || (ui<0.0 && pmax[i]<= par[i] + delta[i] + small)
      if !swept[i] && violation
        sweep!(tableau, i, false)
        swept[i] = true
        cycle_main_loop = true
        break
      end
    end
    if cycle_main_loop; continue; end
    return (iteration, delta)
  end
  return (0, delta)
end # function quadratic_program

"""
Sweeps or inverse sweeps the symmetric tableau A on its kth diagonal entry.
"""
function sweep!(matrix_a::Matrix{Float64}, k::Int, inverse::Bool = false)

  p = 1.0 / matrix_a[k, k]
  v = matrix_a[:, k]
  matrix_a[:, k] .= 0.0
  matrix_a[k, :] .= 0.0
  if inverse
    v[k] = 1.0
  else
    v[k] = -1.0
  end
  for i = 1:size(matrix_a, 1)
    pv = p * v[i]
    matrix_a[:, i] = matrix_a[:, i] - pv * v
  end
end # function sweep!

"""
Checks for constraint violations. The returned value conveys the 
nature of the violation.
"""
function check_constraints(constraint::Matrix{Float64},
  level::Vector{Float64}, pname::Vector{String}, par::Vector{Float64},
  pmin::Vector{Float64}, pmax::Vector{Float64}, io::IO, travel::String)

  tol = 1e-4
  error = false
  (constraints, pars) = size(constraint)
  #
  # Check for bound violations.
  #
  for i = 1:pars
    if par[i] < pmin[i]
      println(io,
        " \nError: Parameter ", i, " is less than its minimum.")
      error = true
    elseif par[i] > pmax[i]
      println(io,
        " \nError: Parameter ", i, " is greater than its maximum.")
      error = true
    end
    if pmin[i] > pmax[i] - tol
      println(io, " \nError: The bounds on parameter ", i,
                  " are too close or inconsistent.")
      error = true
    end
  end
  #
  # Check for constraint violations.
  #
  if constraints > 0
    y = constraint * par
    for i = 1:constraints
      if abs(y[i] - level[i]) > tol
        println(io,
          " \nError: Linear equality constraint ", i, " is not satisfied.")
        error = true
      end
      if count(!iszero, constraint[i, :]) != 1
        continue
      else
        for j = 1:pars
          if constraint[i, j] != 0.0 &&
          (abs(par[j] - pmin[j]) < tol || abs(par[j] - pmax[j]) < tol)
            println(io,
            " \nError: parameter ", j, " is constrained to one of its bounds.")
            error = true
          end
        end
      end
    end
    if rank(constraint) < size(constraint, 1)
      println(io, " \nError: Some equality constraints are redundant.")
      error = true
    end
  end
  #
  # Echo the bounds and the constraints.
  #
  if travel == "search"
    println(io, " \nParameter minima and maxima:")
    name = join(pname, "    ")
    println(io, " \n    ", name)
    if pmin[1] == -Inf
      @printf(io, " \n    -Inf     ")
    else
      @printf(io, " \n%12.4e", pmin[1])
    end
    for i = 2:pars
      if pmin[i] == -Inf
        @printf(io, "   -Inf     ")
      else
        @printf(io, "%12.4e", pmin[i])
      end
    end
    if pmax[1] == Inf
      @printf(io, " \n     Inf     ")
    else
      @printf(io, " \n%12.4e", pmax[1])
    end
    for i = 2:pars
      if pmax[i] == Inf
        @printf(io, "    Inf     ")
      else
        @printf(io, "%12.4e", pmax[i])
      end
    end
    println(io, " ")
    if constraints > 0
      println(io, " \nParameter constraints:")
      println(io, " \n    ", name, "    level ")
      println(io, "")
      for i = 1:constraints
        for j = 1:pars
          @printf(io, "%12.4e", constraint[i, j])
        end
        @printf(io, "%12.4e\n", level[i])
      end
    end
  end
  return error
end # function check_constraints

"""
Outputs the current iteration.
"""
function iteration_output!(io::IO, f::Float64, iteration::Int,
  par::Vector{Float64}, pname::Vector{String}, steps::Int = 0)

  if iteration == 1
    name = join(pname, "    ")
    println(io, " \niter  steps   function    ", name)
  end
  @printf(io, " \n %2i   %2i   %11.4e ", iteration, steps, f)
  for i = 1:length(par)
     @printf(io, "%12.4e", par[i])
  end
  println(io, " ")
end # function iteration_output!

"""
Computes the asymptotic covariance matrix of the
parameter estimates.
"""
function asymptotic_covariances(io::IO, constraint::Matrix{Float64},
  boundary::BitArray{1}, d2f::Matrix{Float64}, pname::Vector{String})
  #
  # Reparameterize to find the asymptotic covariance matrix.
  # Add an extra constraint for each parameter occurring on a boundary.
  #
  (constraints, pars) = size(constraint)
  asy_cov = zeros(pars, pars)
  asy_cov[1:constraints, :] = constraint
  j = constraints + 1
  for i = 1:pars
    if boundary[i]
      asy_cov[j, i] = 0.0
      j = j + 1
    end
  end
  (matrix_u, d, matrix_v) = svd(asy_cov, full = true)
  ker = matrix_v[:, j:pars]
  matrix = ker' * d2f * ker
  asy_cov = ker * inv(matrix) * ker'
  #
  # Standardize the asymptotic covariance matrix.
  #
  for i = 1:pars
    if asy_cov[i, i] <= 1e-10
      asy_cov[i, i] = 0.0
    else
      asy_cov[i, i] = sqrt(asy_cov[i, i])
    end
    for j = 1:i - 1
      if asy_cov[i, i] > 1e-5 &&  asy_cov[j, j] > 1e-5
        asy_cov[i, j] = asy_cov[i, j] / (asy_cov[i, i] * asy_cov[j, j])
      else
        asy_cov[i, j] = 0.0
        asy_cov[j, i] = 0.0
      end
    end
  end
  #
  # Output the results.
  #
  println(io, " \nThe asymptotic standard errors of the parameters:")
  name = join(pname, "    ")
  println(io, " \n    ", name)
  println(io, " ")
  for i = 1:pars
    @printf(io, "%12.4e", asy_cov[i, i])
  end
  println(io, "")
  println(io, " \nThe asymptotic correlation matrix of the parameters:")
  println(io, " \n    ", name)
  println(io, "")
  @printf(io, "%10.4f\n", 1.0)
  for i = 2:pars
    println(io, "")
    @printf(io, "%10.4f", asy_cov[i, 1])
    for j = 2:i - 1
      @printf(io, "%12.4f", asy_cov[i, j])
    end
    if asy_cov[i, i] > 1e-5
      @printf(io, "%12.4f", 1.0)
    else
      @printf(io, "%12.4f", 0.0)
    end
    println(io, "")
  end
end # function asymptotic_covariances

"""Sets defaults for Search."""
function set_search_defaults(constraints, pars, points, travel)
  #
  # This function provides the defaults in parameter initialization.
  #
  # Check that input variables are properly defined.
  #
  if constraints < 0
    throw(ArgumentError("The number of constraints must be nonnegative.")) 
  end
  if pars < 0
    throw(ArgumentError("The number of parameters must be nonnegative.")) 
  end
  if points < 0
    throw(ArgumentError("The number of grid points must be nonnegative.")) 
  end
  if travel != "grid" && travel != "search"
    throw(ArgumentError("The variable travel must be grid or search.")) 
  end
  #
  # Set defaults for arrays.
  #
  grid = zeros(points,pars)
  pname = ["par" for i = 1:pars]
  for i = 1:pars
    pname[i] = pname[i]*" $i"
    pname[i] = rpad(pname[i], 8, ' ')
  end
  par = zeros(pars)
  pmin = zeros(pars)
  pmin[1:pars] .= -Inf
  pmax = zeros(pars)
  pmax[1:pars] .= Inf
  constraint = zeros(constraints, pars)
  level = zeros(constraints)
  goal = "minimize"
  return(grid, pname, par, pmin, pmax, constraint, level, goal)
end

function initial1()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (0, 2, 1, "search")
  (title, cases, standard_errors) = ("test problem 1", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = -2.0
  par[2] = 1.0
  pmin[2] = -3.0 / 2.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun1(par)
#
# Define a function to be minimized. 
#
  f = 100 * (par[2] - par[1]^2)^2 + (1 - par[1])^2
  return f
end

function initial2()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (0, 2, 1, "search")
  (title, cases, standard_errors) = ("test problem 2", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 10.0
  par[2] = 1.0
  pmin[2] = 0.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun2(par)
#
# Define a function to be minimized. 
#
  f = par[2] + (1e-5) * (par[2] - par[1])^2
  return f
end

function initial3()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (0, 2, 1, "search")
  (title,cases,standard_errors) = ("test problem 3", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 1.125
  par[2] = .125
  pmin[1] = 1.0
  pmin[2] = 0.
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun3(par)
#
# Define a function to be minimized. 
#
  f = (par[1] + 1)^3/3.0 + par[2]
  return f
end

function initial4()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (0, 2, 1, "search")
  (title, cases, standard_errors) = ("test problem 4", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 0.
  par[2] = 0.
  pmax[1] = 4.0
  pmax[2] = 3.0
  pmin[1] = -1.5
  pmin[2] = -3.0
  return (grid, pname, par, pmin, pmax, constraint, level,
    travel, goal, title, cases, standard_errors)
end

function fun4(par)
#
# Define a function to be minimized. 
#
  f = sin(par[1] + par[2]) + (par[1] - par[2])^2 -1.5 * par[1] + 2.5*par[2]
  return f
end

function initial5()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (1, 2, 1, "search")
  (title, cases, standard_errors) = ("test problem 5", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 0.
  par[2] = 0.
  constraint[1,1] = 4.
  constraint[1,2] = -3.0
  return (grid, pname, par, pmin, pmax, constraint, level,
    travel, goal, title, cases, standard_errors)
end

function fun5(par)
#
# Define a function to be minimized. 
#
  f = sin(pi * par[1] / 12.0) * cos(pi * par[2] / 16.)
  return f
end

function initial6()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (1, 3, 1, "search")
  (title, cases, standard_errors) = ("test problem 6", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 3.0 
  par[2] = 1.0
  par[3] = 29.
  pmax[1] = 50.
  pmax[2] = 50.
  pmin[1] = 2.0
  pmin[2] = -50.
  pmin[3] = 10.
  constraint[1,1] = 10.
  constraint[1,2] = -1.0
  constraint[1,3] = -1.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel,
    goal, title, cases, standard_errors)
end

function fun6(par)
#
# Define a function to be minimized. 
# 
  f = .01 * par[1]^2 + par[2]^2 - 100.0
  return f
end

function initial7()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (1, 3, 1, "search")
  (title, cases, standard_errors) = ("test problem 7", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = -4.
  par[2] = 1.0
  par[3] = 1.0
  constraint[1,1] = 1.0
  constraint[1,2] = 2.0
  constraint[1,3] = 3.0
  level[1] = 1.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun7(par)
#
# Define a function to be minimized. 
#
  f = (par[1] + par[2])^2 + (par[2] + par[3])^2
  return f
end

function initial8()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (1, 4, 1, "search")
  (title, cases, standard_errors) = ("test problem 8", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 10.
  par[2] = 10.
  par[3] = 10.
  par[4] = 50.
  pmax[1] = 20.
  pmax[2] = 11.
  pmax[3] = 42.
  pmax[4] = 72.
  pmin[1] = 0.
  pmin[2] = 0.
  pmin[3] = 0.
  constraint[1,1] = 1.0
  constraint[1,2] = 2.0
  constraint[1,3] = 2.0
  constraint[1,4] = -1.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun8(par)
#
# Define a function to be minimized. 
#
  f = -par[1] * par[2] * par[3]
  return f
end

function initial9()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (2, 4, 1, "search")
  (title, cases, standard_errors) = ("test problem 9", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 1.0
  par[2] = .5
  par[3] = par[1]/3.0^(.5)-par[2]
  par[4] = par[1]+3.0^(.5)*par[2]
  pmax[4] = 6.
  pmin[1] = 0.
  pmin[2] = 0.
  pmin[3] = 0.
  pmin[4] = 0.
  constraint[1,1] = 1.0/3.0^(.5)
  constraint[1,2] = -1.0
  constraint[1,3] = -1.0
  constraint[2,1] = 1.0
  constraint[2,2] = 3.0^(.5)
  constraint[2,4] = -1.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun9(par)
#
# Define a function to be minimized. 
#
  f = ((par[1] - 3.0)^2 - 9.) * par[2]^3
  return f
end

function initial10()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (0, 4, 1, "search")
  (title, cases, standard_errors) = ("test problem 10", 0, false)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = -3.0
  par[2] = -1.0
  par[3] = -3.0
  par[4] = -1.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun10(par)
#
# Define a function to be minimized. 
#
  f = 100.0 * (par[2] - par[1]^2)^2 + (1.0 - par[1])^2+
	   90.0 * (par[4] - par[3]^2)^2 + (1.0 - par[3])^2+ 
	   10.1 * ((par[2] - 1.0)^2 + (par[4] - 1.0)^2)+
	   19.8 * (par[2] - 1.0) * (par[4] - 1.0)
  return f # (f,df,Nothing)
end

function initial11()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (0, 4, 1, "search")
  (title, cases, standard_errors) = ("nonlinear least squares",21,true)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  par[1] = 10.
  par[2] = -0.1
  par[3] = 5.
  par[4] = -0.01
  pmax[2] = 0.
  pmax[4] = 0.
  return (grid, pname, par, pmin, pmax, constraint, level,
    travel, goal, title, cases, standard_errors) 
end

function fun11(par)
#
# Define a function to be minimized. 
#
  count = [15.1117,11.3601,9.7652,9.0935,8.4820,7.6891,7.3342,
           7.0593,6.7041,6.4313,6.1554,5.9940,5.7698,5.6440,5.3915,5.0938,
           4.8717,4.5996,4.4968,4.3602,4.2668]
  weight = [.004379,.007749,.010487,.012093,.013900,.016914,
           .018591,.020067,.022249,.024177,.026393,.027833,.030039,.031392,
           .034402,.038540,.042135,.047267,.049453,.052600,.054928]
  time = [2.,4.,6.,8.,10.,15.,20.,25.,30.,40.,50.,60.,70.,80.,
           90.,110.,130.,150.,160.,170.,180.]
#
# Exponential fitting.
#
  f = 0.0
  for i = 1:length(count)
    g = par[1] * exp(par[2] * time[i]) + par[3] * exp(par[4] * time[i])
    residual = count[i] - g
    f = f + weight[i] * residual^2
  end
  return f 
end

function initial12()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (1, 3, 1, "search")
  (title, cases, standard_errors) = ("ABO frequency estimation", 0, true)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  fill!(pmin,1e-6)
  fill!(par,1/3)
  fill!(constraint, 1.0)
  fill!(level, 1.0)
  pname[1] = "A freq  "
  pname[2] = "B freq  "
  pname[3] = "O freq  "
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun12(par)
#
# Define a function to be minimized. 
#
  count = [182.,60.,17.,176.]
#
# This problem estimates ABO allele frequencies.
#
  f = 0.0
  for i = 1:4
    if i == 1
      q = par[1]^2 + 2par[1] * par[3]
    elseif i == 2
      q = par[2]^2 + 2par[2] * par[3]
    elseif i == 3
      q = 2par[1] * par[2]
    else
      q = par[3]^2
    end
    f = f - count[i] * log(q)
  end
  return f 
end

function initial13()
#
# This function initializes a minimization problem.
# First set scalar constants.
#
  (constraints, pars, points, travel) = (1, 2, 1, "search")
  (title, cases, standard_errors) = ("survival data", 0, true)
#
#  Set defaults for arrays.
#
  (grid, pname, par, pmin, pmax, constraint, level, goal) =
    set_search_defaults(constraints, pars, points, travel)
#
#  Change these defaults as needed.
#
  fill!(pmin, 1e-6)
  fill!(par, 10.0)
  pname[1] = "mean 1  "
  pname[2] = "mean 2  "
  constraint[1,1] = 1.0
  constraint[1,2] = -1.0
  return (grid, pname, par, pmin, pmax, constraint, level, travel, 
    goal, title, cases, standard_errors)
end

function fun13(par)
#
# Define a function to be minimized. 
#
  time = [6.,6.,6.,7.,10.,13.,16.,22.,23.,6.,9.,10.,11.,17.,
          19.,20.,25.,32.,32.,34.,35.,1.,1.,2.,2.,3.,4.,4.,5.,
          5.,8.,8.,8.,8.,11.,11.,12.,12.,15.,17.,22.,23.]
  died = zeros(Int64,42)
  died[10:21] .= 1
  group = ones(Int64, 42)
  group[22:42] .= 2
#
   pars = length(par)
   f = 0.0
#
# Loop over all cases.
#
  for j = 1:length(time)
    i = group[j]
    mean = par[i]
    if died[j] == 1
       g = exp(-time[j] / mean)
       f = f - log(g)
    else
       g = mean
       f = f + log(g) + time[j] / g
    end
  end
  return f 
end

#
# Run the various test problems.
#
outfile = "search.out"
io = open(outfile, "w")
(par, f) = search(fun1, initial1, io)
(par, f) = search(fun2, initial2, io)
(par, f) = search(fun3, initial3, io)
(par, f) = search(fun4, initial4, io)
(par, f) = search(fun5, initial5, io)
(par, f) = search(fun6, initial6, io)
(par, f) = search(fun7, initial7, io)
(par, f) = search(fun8, initial8, io)
(par, f) = search(fun9, initial9, io)
(par, f) = search(fun10, initial10, io)
(par, f) = search(fun11, initial11, io)
(par, f) = search(fun12, initial12, io)
(par, f) = search(fun13, initial13, io)
close(io)
