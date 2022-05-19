
function rollout!(solver::iLQRSolver{T,Q,n}, α) where {T,Q,n}
    Z = solver.Z; Z̄ = solver.Z̄
    K = solver.K; d = solver.d;
    # update x0
    # # B0, d0, _x0 = solver.B0, solver.d0, solver.x0
    # # n̂, m0 = size(B0)    
    # # x0 = ([(I-B0*B0') * _x0[1:n̂] + B0 * _x0[n̂+1:end]; _x0[n̂+1:end]] # 0th step dynamics
    # #         .+ [B0 * d0; d0] .* α) # 0th step controls
    x0 = solver.x0

    # Start regular roll out
    Z̄[1].z = [x0; control(Z[1])]
    temp = 0.0
	δx = solver.S[end].q
	δu = solver.S[end].r

    for k = 1:solver.N-1
        δx .= RobotDynamics.state_diff(solver.model, state(Z̄[k]), state(Z[k]))
		δu .= d[k] .* α
		mul!(δu, K[k], δx, 1.0, 1.0)
        ū = control(Z[k]) + δu
        RobotDynamics.set_control!(Z̄[k], ū)
        
        # Z̄[k].z = [state(Z̄[k]); control(Z[k]) + δu]
        Z̄[k+1].z = [RobotDynamics.discrete_dynamics(Q, solver.model, Z̄[k]);
        control(Z[k+1])]

        max_x = norm(state(Z̄[k+1]),Inf)
        if max_x > solver.opts.max_state_value || isnan(max_x)
            solver.stats.status = STATE_LIMIT
            # println(k, " : ", state(Z̄[k]))
            return false
        end
        max_u = norm(control(Z̄[k+1]),Inf)
        if max_u > solver.opts.max_control_value || isnan(max_u)
            solver.stats.status = CONTROL_LIMIT 
            return false
        end
    end
    solver.stats.status = UNSOLVED
    return true
end

"Simulate the forward the dynamics open-loop"
function rollout!(solver::iLQRSolver{<:Any,Q}) where Q
    rollout!(Q, solver.model, solver.Z, SVector(solver.x0))
    for k in eachindex(solver.Z)
        solver.Z̄[k].t = solver.Z[k].t
    end
end
