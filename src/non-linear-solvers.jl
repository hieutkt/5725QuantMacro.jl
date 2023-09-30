export
    bisection_solve_f, dekker_solve_f, brent_solve_f

"""Bisection algorithm to to find root of function f, between [a, b]"""
function bisection_solve_f(f::Function, a, b; ε=1e-7, max_iter=1e2)
    f_a = f(a)
    f_b = f(b)
    # Make sure f(a) and f(b) have opposite signs
    @assert f_a*f_b < 0 "With f(a) = "*string(f_a)*"& f(b) = "*string(f_b)*": Not a proper bracket!!"
    # Compute the mid point
    m = (a + b) / 2
    f_m = f(m)
    # Initialize iterating
    iter = 0
    while abs(f_m) >= ε && iter <= max_iter
        iter += 1
        f_m*f_b >= 0 ? b = m : a = m
        m = (a + b) / 2
        f_m = f(m)
    end
    return m
end


"""Dekker's (1969) method to find root of function f, between [a, b]"""
function dekker_solve_f(f::Function, a, b; ε=1e-7, max_iter=1e2)
    # Initiation
    f_a = f(a)
    f_b = f(b)
    # Make sure f(a) and f(b) have opposite signs
    @assert f_a*f_b < 0 "With f(a) = "*string(f_a)*"& f(b) = "*string(f_b)*": Not a proper bracket!!"
    # Make sure b is the 'better' guess than a
    abs(f_b) > abs(f_a) ? (a, f_a, b, f_b) = (b, f_b, a, f_a) : nothing
    # Set the last iteration of b to a
    c, f_c = a, f_a
    # Initialize iterating
    iter = 0
    while abs(f_b) >= ε && iter <= max_iter
        iter += 1
        # Compute the mid point
        m = (a + b) / 2
        # Compute the secant point; fall back to m in case s undefined
        f_b == f_c ? s = m : s = b - f_b * (b-c)/(f_b-f_c)
        # Update values:
        # b becomes c, the last iterations' best guess
        c, f_c = b, f_b
        # If s is between m and b => s becomes new b, else take m
        m < s < b || b < s < m ? b = s : b = m
        f_b = f(b)
        # If the b changes sign, assign the previous iteration as a
        f_b*f_c < 0 ? (a, f_a) = (c, f_c) : nothing
        # Make sure b is the 'better' guess than a
        abs(f_b) > abs(f_a) ? (a, f_a, b, f_b) = (b, f_b, a, f_a) : nothing
    end
    return b
end


"""Brent's (1973) method to find root of function f, between [a, b]"""
function brent_solve_f(f::Function, a, b; ε=1e-7, max_iter=1e2)
    # Initiation
    f_a = f(a)
    f_b = f(b)
    # Make sure f(a) and f(b) have opposite signs
    @assert f_a*f_b < 0 "With f(a) = "*string(f_a)*"& f(b) = "*string(f_b)*": Not a proper bracket!!"
    # Make sure b is the 'better' guess than a
    abs(f_b) > abs(f_a) ? (a, f_a, b, f_b) = (b, f_b, a, f_a) : nothing
    # Set the last iteration of b to a
    d, f_d = c, f_c = a, f_a
    # This flags if the previous iteration used a bisection
    mflag = true
    # Initialize iterating
    iter = 0
    while abs(f_b) >= ε && f_b != 0 && iter <= max_iter
        iter += 1
        if f_a != f_c && f_b != f_c && f_a != f_c
            # Inverse quadratic interpolation
            s = a*f_b*f_c/((f_a-f_b)*(f_a-f_c)) + b*f_a*f_c/((f_b-f_a)*(f_b-f_c)) + c*f_a*f_b/((f_c-f_a)*(f_c-f_b))
        else
            # Compute the secant point; fall back to m in case s undefined
            f_b == f_c ? s = (a+b)/2 : s = b - f_b * (b - c) / (f_b - f_c)
        end
        if  !((3a+b)/4 < s < b || b < s < (3a+b)/4)||
            (mflag  && ( abs(s-b) >= abs(b-c)/2 )) ||
            (!mflag && ( abs(s-b) >= abs(c-d)/2 )) ||
            (mflag  && ( abs(b-c) < ε ))           ||
            (!mflag && ( abs(c-d) < ε ))
            # bisection
            s = (a+b)/2
            mflag = true
        else
            mflag = false
        end
        f_s = f(s)
        d, f_d = c, f_c
        c, f_c = b, f_b
        b, f_b = s, f_s
        # If the b changes sign, assign the previous iteration as a
        f_b*f_c < 0 ? (a, f_a) = (c, f_c) : nothing
        # Make sure b is the 'better' guess than a
        abs(f_b) > abs(f_a) ? (a, f_a, b, f_b) = (b, f_b, a, f_a) : nothing
    end
    return b
end
