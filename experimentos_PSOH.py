import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from poliastro.maneuver import Maneuver
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

ORB_I = Orbit.circular(Earth, alt=400 * u.km)
R_F      = 42164e3
R_I      = ORB_I.a
R_F_FAR  = 135562732.0

def fitness_function(rho, r_f, rho_max):
    if rho < 1.0 or rho > rho_max:
        return 1e6

    r_b = rho * r_f
    try:
        man = Maneuver.bielliptic(ORB_I, r_b * u.m, r_f * u.m)
        return man.get_total_cost().to(u.m / u.s).value
    except:
        return 1e6


def get_local_best(personal_best_pos, personal_best_cost, neighborhood_size):
    n = len(personal_best_pos)
    local_best_pos = np.empty(n)

    for i in range(n):
        # Wrap-around neighbour indices (ring topology)
        neighbours = [(i + k) % n for k in range(-neighborhood_size, neighborhood_size + 1)]
        best_neighbour = min(neighbours, key=lambda idx: personal_best_cost[idx])
        local_best_pos[i] = personal_best_pos[best_neighbour]

    return local_best_pos

def mutate(velocity):
    if np.random.random() < 0.1:
        return velocity * (np.random.random() + 0.5)
    else:
        return velocity


def pso_local(
    r_f,
    rho_max,
    n_particles: int = 30,
    n_iterations: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    neighborhood_size: int = 2,   # neighbours on each side → ring of 2k+1 particles
    #seed: int = 42,
):
    #rng = np.random.default_rng(seed)

    lower = 1
    upper = rho_max

    # --- Initialisation ---
    positions  = np.random.uniform(lower, upper, n_particles)
    print(positions)
    velocities = np.random.uniform(-(upper - lower), (upper - lower), n_particles)

    personal_best_pos  = positions.copy()
    personal_best_cost = np.array([fitness_function(p, r_f, rho_max) for p in positions])

    # Track overall best just for reporting
    global_best_idx  = np.argmin(personal_best_cost)
    global_best_pos  = personal_best_pos[global_best_idx]
    global_best_cost = personal_best_cost[global_best_idx]

    history = [global_best_cost]

    # --- Main loop ---
    for iteration in range(n_iterations):
        # Compute local best position for every particle
        local_best_pos = get_local_best(personal_best_pos, personal_best_cost, neighborhood_size)

        r1 = np.random.random(n_particles)
        r2 = np.random.random(n_particles)

        # Evaluate fitness
        costs = np.array([fitness_function(p, r_f, rho_max) for p in positions])

        # Update personal bests
        improved = costs < personal_best_cost
        personal_best_pos[improved]  = positions[improved]
        personal_best_cost[improved] = costs[improved]

        # Update global best (tracking only)
        best_idx = np.argmin(costs)
        global_best_pos = positions[best_idx]
        global_best_cost = np.min(costs)

        history.append(global_best_cost)

        # Velocity update — social term now pulls toward local best, not global best
        velocities = (
            w  * velocities
            + c1 * r1 * (personal_best_pos - positions)
            + c2 * r2 * (local_best_pos     - positions)
        )

        velocities = np.array([mutate(v) for v in velocities])

        positions = positions + velocities

        print(f"Iteration {iteration + 1:4d}/{n_iterations} | "
              f"Best rho: {global_best_pos:.6f} | "
              f"Delta-v: {global_best_cost:.4f} m/s")

    return global_best_pos, global_best_cost, history

def pso(
    r_f,
    rho_max,
    n_particles: int = 30,
    n_iterations: int = 70,
    w: float = 0.7,        # inertia weight
    c1: float = 1.5,       # cognitive (personal best) coefficient
    c2: float = 1.5,       # social (global best) coefficient
    #seed: int = 42,
):
    #rng = np.random.default_rng(seed)

    lower = 1
    upper = rho_max

    # --- Initialisation ---
    positions  = np.random.uniform(lower, upper, n_particles)
    print(positions)
    velocities = np.random.uniform(-(upper - lower), (upper - lower), n_particles)

    personal_best_pos  = positions.copy()
    personal_best_cost = np.array([fitness_function(p, r_f, rho_max) for p in positions])

    global_best_idx  = np.argmin(personal_best_cost)
    global_best_pos  = personal_best_pos[global_best_idx]
    global_best_cost = personal_best_cost[global_best_idx]

    history = [global_best_cost]

    # --- Main loop ---
    for iteration in range(n_iterations):
        r1 = np.random.random(n_particles)
        r2 = np.random.random(n_particles)

        # Velocity update
        velocities = (
            w  * velocities
            + c1 * r1 * (personal_best_pos - positions)
            + c2 * r2 * (global_best_pos   - positions)
        )

        # Evaluate fitness
        costs = np.array([fitness_function(p, r_f, rho_max) for p in positions])

        # Update personal bests
        improved = costs < personal_best_cost
        personal_best_pos[improved]  = positions[improved]
        personal_best_cost[improved] = costs[improved]

        # Update global best
        best_idx = np.argmin(personal_best_cost)
        if personal_best_cost[best_idx] < global_best_cost:
            global_best_pos  = personal_best_pos[best_idx]
            global_best_cost = personal_best_cost[best_idx]

        temp1 = np.argmin(costs)
        temp2 = personal_best_cost[temp1]
        temp3 = personal_best_pos[temp1]

        history.append(temp2)

        velocities = np.array([mutate(v) for v in velocities])

        positions = positions + velocities

        print(f"Iteration {iteration + 1:4d}/{n_iterations} | "
              f"Best rho: {temp3:.6f} | "
              f"Delta-v: {temp2:.4f} m/s")

    return global_best_pos, global_best_cost, history

# --- Run ---
if __name__ == "__main__":
    r_f_loc = R_F
    #r_f_loc = R_F_FAR

    best_rho, best_cost, history = pso_local(
        r_f=r_f_loc,
        rho_max=40,
        n_particles=30,
        n_iterations=70,
        neighborhood_size=2,   # each particle sees 5 neighbours total (itself ± 2)
    )

    ga_man = Maneuver.bielliptic(ORB_I, best_rho * r_f_loc * u.m, r_f_loc * u.m)

    print(f"\nOptimal rho  : {best_rho:.6f}")
    print(f"Optimal r_b  : {best_rho * r_f_loc / 1e3:.2f} km")
    print(f"Min delta-v  : {best_cost:.4f} m/s")
    print(f"Duration  : {ga_man.get_total_time().to(u.day).value:.4f} days")

    h_man1 = Maneuver.hohmann(ORB_I, 42164 * u.km)
    dv_hohmann1 = h_man1.get_total_cost().to(u.m / u.s).value

    #h_man2 = Maneuver.hohmann(ORB_I, R_F_FAR * u.m)
    #dv_hohmann2 = h_man2.get_total_cost().to(u.m / u.s).value

    plt.figure(figsize=(10, 6))

    generations = range(len(history))
    plt.plot(generations, history, label="GA - Cenário 1 (GEO)", color='blue', marker='o', markersize=3)
    #plt.plot(generations, history, label="GA - Cenário 2 (Distante)", color='green', marker='x', markersize=3)


    #plt.axhline(y=dv_hohmann1, color='blue', linestyle='--', alpha=0.6, label=f"Ref. Hohmann GEO ({dv_hohmann1:.1f} m/s)")
    #plt.axhline(y=dv_hohmann2, color='green', linestyle='--', alpha=0.6, label=f"Ref. Hohmann Distante ({dv_hohmann2:.1f} m/s)")

    plt.xlabel("Iteração")
    plt.ylabel("Delta-V Mínimo (m/s)")
    plt.title("Convergência do PSO Híbrido")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Opcional: Ajustar limites para ver melhor a convergência se os valores forem muito distantes
    # plt.ylim(min(log_far + log_geo) - 100, max(log_far + log_geo) + 100)

    plt.show()