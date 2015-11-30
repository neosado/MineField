# Author: Youngjun Kim, youngjun@stanford.edu
# Date: 10/15/2015

module Dist_

import Base: rand, mean, maximum, minimum, string

export FixedValue, RareDist


using Distributions


type FixedValue

    v::Union{Int64, Float64}
end

function rand(D::FixedValue)

    return D.v
end

function mean(D::FixedValue)

    return D.v
end

function maximum(D::FixedValue)

    return D.v
end

function minimum(D::FixedValue)

    return D.v
end

function string(D::FixedValue)

    return "FixedValue(" * string(D.v) * ")"
end


type RareDist
    
    p::Float64
    r_type::Symbol
    r::Float64
    Dr::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution}
    D::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution}

    function RareDist(p::Float64, r::Float64, D::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution})
        
        self = new()
        
        self.p = p
        self.r_type = :real
        self.r = r
        self.D = D
        
        return self
    end

    function RareDist(p::Float64, Dr::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution}, D::Union{DiscreteUnivariateDistribution, ContinuousUnivariateDistribution})
        
        self = new()
        
        self.p = p
        self.r_type = :dist
        self.Dr = Dr
        self.D = D
        
        return self
    end
end

function rand(D::RareDist)
    
    u = rand()
    
    if u < D.p
        if D.r_type == :real
            return D.r
        elseif D.r_type == :dist
            return rand(D.Dr)
        end
    else
        return rand(D.D)
    end
end

function mean(D::RareDist)

    if D.r_type == :real
        return D.p * D.r + (1 - D.p) * mean(D.D)
    elseif D.r_type == :dist
        return D.p * mean(D.Dr) + (1 - D.p) * mean(D.D)
    end
end

function maximum(D::RareDist)

    if D.p == 0.
        return maximum(D.D)
    else
        if D.r_type == :real
            return max(D.r, maximum(D.D))
        elseif D.r_type == :dist
            return max(maximum(D.Dr), maximum(D.D))
        end
    end
end

function minimum(D::RareDist)

    if D.p == 0.
        return minimum(D.D)
    else
        if D.r_type == :real
            return min(D.r, minimum(D.D))
        elseif D.r_type == :dist
            return min(minimum(D.Dr), minimum(D.D))
        end
    end
end

function string(D::RareDist)

    if D.p == 0.
        return "RareDist(" * string(D.D) * ")"
    else
        if D.r_type == :real
            return "RareDist(" * string(D.p) * ", " * string(D.r) * "; " * string(1 - D.p) * ", " * string(D.D) * ")"
        elseif D.r_type == :dist
            return "RareDist(" * string(D.p) * ", " * string(D.Dr) * "; " * string(1 - D.p) * ", " * string(D.D) * ")"
        end
    end
end


end


