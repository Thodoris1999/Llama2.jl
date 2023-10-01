struct LoraDenseOp{T,N}
    weights::Array{T,N}
    lora_A_weights::Any
    lora_B_weights::Any
    input::Tensor{T,2}
    lora_A_out::Tensor{T,2}
    output::Tensor{T,2}
end

function lora_dense(input::Tensor{T,2}; out_features::Int, rank::Int, name::String) where {T}
    graph = input.graph
    in_features = size(input, 1)
    batch_size = size(input, 2)

    lora_A_out = Tensor((
        rank,
        batch_size,
    ), graph)
    output = Tensor((
        out_features,
        batch_size,
    ), graph)

    weights = zeros(T, (in_features, out_features))
    graph.parameter_dict[name] = weights
    lora_A_weights_name = string("_lora_a_", name)
    lora_A_weights = Tensor((
        in_features,
        rank,
    ), graph; parameter=true, name=lora_A_weights_name)
    lora_B_weights_name = string("_lora_b_", name)
    lora_B_weights = Tensor((
        rank,
        out_features,
    ), graph; parameter=true, name=lora_B_weights_name)

    k = 1 / in_features
    rand!(Uniform(-sqrt(k), sqrt(k)), lora_A_weights.value)
    fill!(lora_B_weights.value, 0)

    push!(graph.operations, LoraDenseOp(weights, lora_A_weights, lora_B_weights, input, lora_A_out, output))
    return output
end

function merge!(op::LoraDenseOp)
    W = op.weights
    A = op.lora_A_weights.value
    B = op.lora_B_weights.value

    @tturbo for i in axes(W, 1), j in axes(W, 2)
        for k in axes(A, 2)
            W[i, j] += A[i, k] * B[k, j]
        end
    end
end

function unmerge!(op::LoraDenseOp)
    W = op.weights
    A = op.lora_A_weights.value
    B = op.lora_B_weights.value

    @tturbo for i in axes(W, 1), j in axes(W, 2)
        for k in axes(A, 2)
            W[i, j] -= A[i, k] * B[k, j]
        end
    end
end

function forward!(op::LoraDenseOp)
    # frozen forward
    W = op.weights
    x = op.input.value
    y = op.output.value

    @tturbo for i in axes(x, 2), m in axes(W, 2)
        s = zero(eltype(y))
        for k in axes(W, 1)
            s += W[k, m] * x[k, i]
        end
        y[m, i] = s
    end

    # lora forward
    W = op.lora_A_weights.value
    x = op.input.value
    y = op.lora_A_out.value

    @tturbo for i in axes(x, 2), m in axes(W, 2)
        s = zero(eltype(y))
        for k in axes(W, 1)
            s += W[k, m] * x[k, i]
        end
        y[m, i] = s
    end

    W = op.lora_B_weights.value
    x = op.lora_A_out.value
    y = op.output.value

    @tturbo for i in axes(x, 2), m in axes(W, 2)
        s = zero(eltype(y))
        for k in axes(W, 1)
            s += W[k, m] * x[k, i]
        end
        y[m, i] += s
    end

    return nothing
end

function backward!(op::LoraDenseOp)
    # LoRA B backward
    W = op.lora_B_weights.value
    ∂W = op.lora_B_weights.grad
    x = op.lora_A_out.value
    ∂x = op.lora_A_out.grad
    ∂y = op.output.grad

    # input gradient
    @tturbo for i in axes(∂y, 2), k in axes(W, 1)
        s = zero(eltype(∂x))
        for m in axes(W, 2)
            s += W[k, m] * ∂y[m, i]
        end
        ∂x[k, i] += s
    end

    # weight gradient
    @tturbo for i in axes(∂y, 2), m in axes(∂W, 2), k in axes(∂W, 1)
        ∂W[k, m] += ∂y[m, i] * x[k, i]
    end

    # LoRA A backward
    W = op.lora_A_weights.value
    ∂W = op.lora_A_weights.grad
    x = op.input.value
    ∂x = op.input.grad
    ∂y = op.lora_A_out.grad

    # input gradient
    @tturbo for i in axes(∂y, 2), k in axes(W, 1)
        s = zero(eltype(∂x))
        for m in axes(W, 2)
            s += W[k, m] * ∂y[m, i]
        end
        ∂x[k, i] += s
    end

    W_full = op.weights
    ∂o = op.output.grad

    # input gradient (frozen weight component)
    @tturbo for i in axes(∂o, 2), k in axes(W_full, 1)
        s = zero(eltype(∂x))
        for m in axes(W_full, 2)
            s += W_full[k, m] * ∂o[m, i]
        end
        ∂x[k, i] += s
    end

    # weight gradient
    @tturbo for i in axes(∂y, 2), m in axes(∂W, 2), k in axes(∂W, 1)
        ∂W[k, m] += ∂y[m, i] * x[k, i]
    end

    return nothing
end
