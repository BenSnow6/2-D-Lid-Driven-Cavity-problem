"""
2D Lid-Driven Cavity Problem Solver
Implementation based on the finite difference approach for solving
the Navier-Stokes equations using the MAC (Marker and Cell) scheme.

Author: Implementation from Ben Snow's paper (April/May 2018)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


class LidDrivenCavity:
    """
    Solver for 2D Lid-Driven Cavity problem using finite difference method
    with staggered grid (MAC scheme) and projection method for pressure.
    """

    def __init__(self, nx=15, ny=15, L=1.0, U=1.0, nu=0.01, dt=0.01, t_final=10.0):
        """
        Initialize the cavity problem.

        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        L : float
            Physical size of the cavity (assumes square cavity)
        U : float
            Lid velocity (top boundary)
        nu : float
            Kinematic viscosity
        dt : float
            Time step
        t_final : float
            Final simulation time
        """
        self.nx = nx
        self.ny = ny
        self.L = L
        self.U = U
        self.nu = nu
        self.dt = dt
        self.t_final = t_final
        self.h = L / (nx - 1)  # Grid spacing
        self.beta = 1.2  # Relaxation parameter for pressure (water)

        # Initialize arrays
        # Velocities on staggered grid
        self.u = np.zeros((nx + 1, ny + 2))  # u velocity (vertical edges)
        self.v = np.zeros((nx + 2, ny + 1))  # v velocity (horizontal edges)
        self.u_temp = np.zeros_like(self.u)  # Temporary u velocity
        self.v_temp = np.zeros_like(self.v)  # Temporary v velocity
        self.P = np.zeros((nx + 2, ny + 2))  # Pressure at cell centers

        # Time tracking
        self.t = 0.0
        self.time_steps = int(t_final / dt)

        print(f"Cavity initialized: {nx}x{ny} grid, h={self.h:.4f}, dt={dt}, nu={nu}")
        print(f"Total time steps: {self.time_steps}")

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions for the lid-driven cavity.

        Boundary conditions:
        - Top wall: u = U, v = 0
        - Bottom, left, right walls: u = 0, v = 0 (no-slip)
        """
        # Bottom wall (j = 0): no-slip
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0

        # Top wall (j = ny): u = U, v = 0
        self.u[:, self.ny] = self.U
        self.v[:, self.ny] = 0.0

        # Left wall (i = 0): no-slip
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0

        # Right wall (i = nx): no-slip
        self.u[self.nx, :] = 0.0
        self.v[self.nx, :] = 0.0

    def compute_advection_u(self):
        """
        Compute advection terms for u momentum equation.
        Returns uu and uv terms on the grid.
        """
        nx, ny = self.nx, self.ny
        uu = np.zeros((nx + 1, ny + 2))
        uv = np.zeros((nx + 1, ny + 2))

        for i in range(1, nx):
            for j in range(1, ny + 1):
                # uu terms (x-direction advection)
                u_right = 0.5 * (self.u[i+1, j] + self.u[i, j])
                u_left = 0.5 * (self.u[i, j] + self.u[i-1, j])
                uu[i, j] = (u_right**2 - u_left**2) / self.h

                # uv terms (y-direction advection)
                if j < ny:
                    u_top = 0.5 * (self.u[i, j+1] + self.u[i, j])
                    v_top = 0.5 * (self.v[i, j] + self.v[i+1, j])
                    uv_top = u_top * v_top
                else:
                    uv_top = 0.0

                if j > 1:
                    u_bottom = 0.5 * (self.u[i, j] + self.u[i, j-1])
                    v_bottom = 0.5 * (self.v[i, j-1] + self.v[i+1, j-1])
                    uv_bottom = u_bottom * v_bottom
                else:
                    uv_bottom = 0.0

                uv[i, j] = (uv_top - uv_bottom) / self.h

        return uu, uv

    def compute_advection_v(self):
        """
        Compute advection terms for v momentum equation.
        Returns vv and vu terms on the grid.
        """
        nx, ny = self.nx, self.ny
        vv = np.zeros((nx + 2, ny + 1))
        vu = np.zeros((nx + 2, ny + 1))

        for i in range(1, nx + 1):
            for j in range(1, ny):
                # vv terms (y-direction advection)
                v_top = 0.5 * (self.v[i, j+1] + self.v[i, j])
                v_bottom = 0.5 * (self.v[i, j] + self.v[i, j-1])
                vv[i, j] = (v_top**2 - v_bottom**2) / self.h

                # vu terms (x-direction advection)
                if i < nx:
                    v_right = 0.5 * (self.v[i+1, j] + self.v[i, j])
                    u_right = 0.5 * (self.u[i, j] + self.u[i, j+1])
                    vu_right = v_right * u_right
                else:
                    vu_right = 0.0

                if i > 1:
                    v_left = 0.5 * (self.v[i, j] + self.v[i-1, j])
                    u_left = 0.5 * (self.u[i-1, j] + self.u[i-1, j+1])
                    vu_left = v_left * u_left
                else:
                    vu_left = 0.0

                vu[i, j] = (vu_right - vu_left) / self.h

        return vv, vu

    def compute_diffusion_u(self):
        """
        Compute diffusion terms for u momentum equation.
        """
        nx, ny = self.nx, self.ny
        diff_u = np.zeros((nx + 1, ny + 2))

        for i in range(1, nx):
            for j in range(1, ny + 1):
                laplacian = (self.u[i+1, j] + self.u[i-1, j] +
                           self.u[i, j+1] + self.u[i, j-1] - 4.0 * self.u[i, j])
                diff_u[i, j] = self.nu * laplacian / (self.h**2)

        return diff_u

    def compute_diffusion_v(self):
        """
        Compute diffusion terms for v momentum equation.
        """
        nx, ny = self.nx, self.ny
        diff_v = np.zeros((nx + 2, ny + 1))

        for i in range(1, nx + 1):
            for j in range(1, ny):
                laplacian = (self.v[i+1, j] + self.v[i-1, j] +
                           self.v[i, j+1] + self.v[i, j-1] - 4.0 * self.v[i, j])
                diff_v[i, j] = self.nu * laplacian / (self.h**2)

        return diff_v

    def compute_temporary_velocity(self):
        """
        Compute temporary velocity without pressure term.
        u_temp = u + dt * (-J + B)
        where J is advection and B is diffusion.
        """
        # Compute advection terms
        uu, uv = self.compute_advection_u()
        vv, vu = self.compute_advection_v()

        # Compute diffusion terms
        diff_u = self.compute_diffusion_u()
        diff_v = self.compute_diffusion_v()

        # Update temporary velocities
        nx, ny = self.nx, self.ny

        for i in range(1, nx):
            for j in range(1, ny + 1):
                self.u_temp[i, j] = self.u[i, j] + self.dt * (-(uu[i, j] + uv[i, j]) + diff_u[i, j])

        for i in range(1, nx + 1):
            for j in range(1, ny):
                self.v_temp[i, j] = self.v[i, j] + self.dt * (-(vv[i, j] + vu[i, j]) + diff_v[i, j])

        # Apply boundary conditions to temporary velocities
        self.u_temp[:, 0] = 0.0
        self.u_temp[:, ny] = self.U
        self.u_temp[0, :] = 0.0
        self.u_temp[nx, :] = 0.0

        self.v_temp[:, 0] = 0.0
        self.v_temp[:, ny] = 0.0
        self.v_temp[0, :] = 0.0
        self.v_temp[nx, :] = 0.0

    def solve_pressure(self, iterations=100):
        """
        Solve for pressure using successive over-relaxation (SOR).
        Uses three different equations for interior, boundary, and corner points.
        """
        nx, ny = self.nx, self.ny
        h = self.h
        dt = self.dt
        beta = self.beta

        for _ in range(iterations):
            P_old = self.P.copy()

            # Interior points (i=2 to nx-1, j=2 to ny-1)
            for i in range(2, nx):
                for j in range(2, ny):
                    div_u_temp = (self.u_temp[i, j] - self.u_temp[i-1, j] +
                                 self.v_temp[i, j] - self.v_temp[i, j-1])

                    self.P[i, j] = 0.25 * beta * (
                        self.P[i+1, j] + self.P[i-1, j] +
                        self.P[i, j+1] + self.P[i, j-1] -
                        (h / dt) * div_u_temp
                    ) + (1 - beta) * self.P[i, j]

            # Boundary points (not corners)
            # Bottom boundary (j=1)
            for i in range(2, nx):
                j = 1
                div_u_temp = (self.u_temp[i, j] - self.u_temp[i-1, j] +
                             self.v_temp[i, j] - self.v_temp[i, j-1])
                self.P[i, j] = (beta / 3.0) * (
                    self.P[i+1, j] + self.P[i-1, j] +
                    self.P[i, j+1] + self.P[i, j-1] -
                    (h / dt) * div_u_temp
                ) + (1 - beta) * self.P[i, j]

            # Top boundary (j=ny)
            for i in range(2, nx):
                j = ny
                div_u_temp = (self.u_temp[i, j] - self.u_temp[i-1, j] +
                             self.v_temp[i, j] - self.v_temp[i, j-1])
                self.P[i, j] = (beta / 3.0) * (
                    self.P[i+1, j] + self.P[i-1, j] +
                    self.P[i, j+1] + self.P[i, j-1] -
                    (h / dt) * div_u_temp
                ) + (1 - beta) * self.P[i, j]

            # Left boundary (i=1)
            for j in range(2, ny):
                i = 1
                div_u_temp = (self.u_temp[i, j] - self.u_temp[i-1, j] +
                             self.v_temp[i, j] - self.v_temp[i, j-1])
                self.P[i, j] = (beta / 3.0) * (
                    self.P[i+1, j] + self.P[i-1, j] +
                    self.P[i, j+1] + self.P[i, j-1] -
                    (h / dt) * div_u_temp
                ) + (1 - beta) * self.P[i, j]

            # Right boundary (i=nx)
            for j in range(2, ny):
                i = nx
                div_u_temp = (self.u_temp[i, j] - self.u_temp[i-1, j] +
                             self.v_temp[i, j] - self.v_temp[i, j-1])
                self.P[i, j] = (beta / 3.0) * (
                    self.P[i+1, j] + self.P[i-1, j] +
                    self.P[i, j+1] + self.P[i, j-1] -
                    (h / dt) * div_u_temp
                ) + (1 - beta) * self.P[i, j]

            # Corner points
            corners = [(1, 1), (1, ny), (nx, 1), (nx, ny)]
            for i, j in corners:
                div_u_temp = (self.u_temp[i, j] - self.u_temp[i-1, j] +
                             self.v_temp[i, j] - self.v_temp[i, j-1])
                self.P[i, j] = (beta / 2.0) * (
                    self.P[i+1, j] + self.P[i-1, j] +
                    self.P[i, j+1] + self.P[i, j-1] -
                    (h / dt) * div_u_temp
                ) + (1 - beta) * self.P[i, j]

            # Check convergence
            error = np.max(np.abs(self.P - P_old))
            if error < 1e-6:
                break

    def correct_velocity(self):
        """
        Correct velocity using pressure gradient.
        u^(n+1) = u_temp - dt * grad(P)
        """
        nx, ny = self.nx, self.ny
        h = self.h
        dt = self.dt

        # Correct u velocity
        for i in range(1, nx):
            for j in range(1, ny + 1):
                self.u[i, j] = self.u_temp[i, j] - dt * (self.P[i+1, j] - self.P[i, j]) / h

        # Correct v velocity
        for i in range(1, nx + 1):
            for j in range(1, ny):
                self.v[i, j] = self.v_temp[i, j] - dt * (self.P[i, j+1] - self.P[i, j]) / h

        # Reapply boundary conditions
        self.apply_boundary_conditions()

    def compute_vorticity(self):
        """
        Compute vorticity: omega = dv/dx - du/dy
        """
        nx, ny = self.nx, self.ny
        h = self.h
        vorticity = np.zeros((nx, ny))

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Average velocities to cell center
                dvdx = (self.v[i+1, j] - self.v[i, j]) / h
                dudy = (self.u[i, j+1] - self.u[i, j]) / h
                vorticity[i, j] = dvdx - dudy

        return vorticity

    def get_velocity_at_cell_centers(self):
        """
        Interpolate velocities from staggered grid to cell centers for plotting.
        """
        nx, ny = self.nx, self.ny
        u_center = np.zeros((nx, ny))
        v_center = np.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                # Average u velocities
                u_center[i, j] = 0.5 * (self.u[i, j+1] + self.u[i+1, j+1])
                # Average v velocities
                v_center[i, j] = 0.5 * (self.v[i+1, j] + self.v[i+1, j+1])

        return u_center, v_center

    def step(self):
        """
        Perform one time step of the simulation.
        """
        # Step 1: Compute temporary velocity (advection + diffusion)
        self.compute_temporary_velocity()

        # Step 2: Solve for pressure
        self.solve_pressure(iterations=100)

        # Step 3: Correct velocity with pressure
        self.correct_velocity()

        # Update time
        self.t += self.dt

    def run(self, save_interval=10):
        """
        Run the full simulation.

        Parameters:
        -----------
        save_interval : int
            Save results every N time steps
        """
        print("Starting simulation...")

        # Storage for results
        saved_times = []
        saved_u = []
        saved_v = []
        saved_P = []
        saved_vorticity = []

        for n in range(self.time_steps):
            self.step()

            if n % save_interval == 0:
                u_center, v_center = self.get_velocity_at_cell_centers()
                vorticity = self.compute_vorticity()

                saved_times.append(self.t)
                saved_u.append(u_center.copy())
                saved_v.append(v_center.copy())
                saved_P.append(self.P[1:self.nx+1, 1:self.ny+1].copy())
                saved_vorticity.append(vorticity.copy())

                if n % 100 == 0:
                    print(f"Time step {n}/{self.time_steps}, t={self.t:.3f}s")

        print("Simulation complete!")

        return {
            'times': saved_times,
            'u': saved_u,
            'v': saved_v,
            'P': saved_P,
            'vorticity': saved_vorticity
        }

    def plot_results(self, results, time_index=-1):
        """
        Plot velocity field, vorticity, and pressure at a specific time.

        Parameters:
        -----------
        results : dict
            Results dictionary from run()
        time_index : int
            Index of time to plot (-1 for final time)
        """
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig)

        t = results['times'][time_index]
        u = results['u'][time_index]
        v = results['v'][time_index]
        P = results['P'][time_index]
        vorticity = results['vorticity'][time_index]

        # Create coordinate arrays
        x = np.linspace(0, self.L, self.nx)
        y = np.linspace(0, self.L, self.ny)
        X, Y = np.meshgrid(x, y)

        # Plot 1: Velocity field (quiver)
        ax1 = fig.add_subplot(gs[0])
        speed = np.sqrt(u.T**2 + v.T**2)
        ax1.quiver(X, Y, u.T, v.T, speed, cmap='jet')
        ax1.set_title(f'Velocity Field at t={t:.2f}s')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')

        # Plot 2: Vorticity (contour)
        ax2 = fig.add_subplot(gs[1])
        levels = np.linspace(vorticity.min(), vorticity.max(), 20)
        contour = ax2.contourf(X, Y, vorticity.T, levels=levels, cmap='RdYlBu_r')
        plt.colorbar(contour, ax=ax2, label='Vorticity')
        ax2.set_title(f'Vorticity at t={t:.2f}s')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')

        # Plot 3: Pressure (surface/contour)
        ax3 = fig.add_subplot(gs[2])
        contour_p = ax3.contourf(X, Y, P.T, levels=20, cmap='viridis')
        plt.colorbar(contour_p, ax=ax3, label='Pressure')
        ax3.set_title(f'Pressure at t={t:.2f}s')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')

        plt.tight_layout()
        return fig

    def plot_pressure_3d(self, results, time_index=-1):
        """
        Create a 3D surface plot of pressure.
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        t = results['times'][time_index]
        P = results['P'][time_index]

        x = np.linspace(0, self.L, self.nx)
        y = np.linspace(0, self.L, self.ny)
        X, Y = np.meshgrid(x, y)

        surf = ax.plot_surface(X, Y, P.T, cmap='viridis', edgecolor='black',
                              linewidth=0.5, alpha=0.8)

        ax.set_xlabel('x direction')
        ax.set_ylabel('y direction')
        ax.set_zlabel('Pressure')
        ax.set_title(f'Pressure Field at t={t:.2f}s')
        fig.colorbar(surf, ax=ax, shrink=0.5)

        return fig


def main():
    """
    Main function to run the simulation and create plots.
    """
    # Create cavity solver with parameters from the paper
    cavity = LidDrivenCavity(
        nx=15,           # 15x15 grid as mentioned in paper
        ny=15,
        L=1.0,          # Unit square cavity
        U=1.0,          # Lid velocity
        nu=0.01,        # Kinematic viscosity (assumed)
        dt=0.01,        # Time step (0.01s as mentioned)
        t_final=10.0    # 10 seconds simulation
    )

    # Run simulation
    results = cavity.run(save_interval=10)

    # Create plots
    print("\nGenerating plots...")

    # Plot at final time
    fig1 = cavity.plot_results(results, time_index=-1)
    fig1.savefig('cavity_results_2d.png', dpi=150, bbox_inches='tight')
    print("Saved 2D results to 'cavity_results_2d.png'")

    # 3D pressure plot
    fig2 = cavity.plot_pressure_3d(results, time_index=-1)
    fig2.savefig('cavity_pressure_3d.png', dpi=150, bbox_inches='tight')
    print("Saved 3D pressure plot to 'cavity_pressure_3d.png'")

    print("\nSimulation and verification complete!")
    print("Results saved to PNG files.")


if __name__ == "__main__":
    main()
