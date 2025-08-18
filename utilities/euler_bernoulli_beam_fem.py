"""
Author: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
Date: 2025-08-12 18:06:32
LastEditors: Fei-JH fei.jinghao.53r@st.kyoto-u.ac.jp
LastEditTime: 2025-08-13 18:38:31
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class BeamAnalysis:
    def __init__(self, L, E, I, rho, A, n_elements):
        """
        Initialize beam properties and matrices.

        Parameters
        ----------
        L : float
            Length of the beam.
        E : float
            Young's modulus.
        I : float
            Moment of inertia.
        rho : float
            Density of the material.
        A : float
            Cross-sectional area.
        n_elements : int
            Number of finite elements.
        """
        self.L = L
        self.E = E
        self.I = I
        self.rho = rho
        self.A = A
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.node_coords = np.linspace(0, L, self.n_nodes)
        self.element_nodes = np.array([(i, i + 1) for i in range(n_elements)])
        self.K_global = np.zeros((2 * self.n_nodes, 2 * self.n_nodes))
        self.M_global = np.zeros((2 * self.n_nodes, 2 * self.n_nodes))
        self.element_length = L / n_elements

    @staticmethod
    def _beam_element_stiffness(E, I, L):
        """
        Compute beam element stiffness matrix.

        Parameters
        ----------
        E : float
            Young's modulus.
        I : float
            Moment of inertia.
        L : float
            Element length.

        Returns
        -------
        np.ndarray
            Element stiffness matrix (4x4).
        """
        coeff = E * I / L**3
        return coeff * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])

    @staticmethod
    def _beam_element_mass(rho, A, L):
        """
        Compute beam element mass matrix.

        Parameters
        ----------
        rho : float
            Density of the material.
        A : float
            Cross-sectional area.
        L : float
            Element length.

        Returns
        -------
        np.ndarray
            Element mass matrix (4x4).
        """
        coeff = rho * A * L / 420
        return coeff * np.array([
            [156, 22*L, 54, -13*L],
            [22*L, 4*L**2, 13*L, -3*L**2],
            [54, 13*L, 156, -22*L],
            [-13*L, -3*L**2, -22*L, 4*L**2]
        ])

    def assemble_matrices(self, dmgfield=None, mass_dmg_power=0):
        """
        Assemble global stiffness and mass matrices with optional damage field.

        Parameters
        ----------
        dmgfield : None, list, or np.ndarray
            Damage field for each element. If None, all elements are undamaged.
        mass_dmg_power : float
            Power factor for mass field damage. Default is 2.
        """
        self.K_global[:] = 0
        self.M_global[:] = 0

        # Automatically set damage field
        if dmgfield is None:
            dmg_arr = np.ones(self.n_elements)
        else:
            dmg_arr = np.asarray(dmgfield)
            if dmg_arr.shape[0] != self.n_elements:
                raise ValueError("Invalid damage field shape")

        K_e_base = self._beam_element_stiffness(self.E, self.I, self.element_length)
        M_e_base = self._beam_element_mass(self.rho, self.A, self.element_length)

        for idx, (n1, n2) in enumerate(self.element_nodes):
            dmg_coeff = dmg_arr[idx]
            mass_coeff = dmg_coeff ** (-mass_dmg_power)
            K_e = K_e_base * dmg_coeff
            M_e = M_e_base * mass_coeff
            indices = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])
            np.add.at(self.K_global, (indices[:, None], indices), K_e)
            np.add.at(self.M_global, (indices[:, None], indices), M_e)

    def apply_BC(self):
        """
        Apply simply supported boundary conditions.
        """
        bc_indices = [0, -2]
        self.K_global[bc_indices, :] = 0
        self.K_global[:, bc_indices] = 0
        self.K_global[bc_indices, bc_indices] = 1
        self.M_global[bc_indices, :] = 0
        self.M_global[:, bc_indices] = 0
        self.M_global[bc_indices, bc_indices] = 1
        # # Optionally add small rotational stiffness to boundary rotational DOFs
        # # Left end θ: index 1; Right end θ: index 2*(n_nodes-1)+1
        # self.K_global[1, 1] += 1e-8
        # self.K_global[-1, -1] += 1e-8

    def solve_eigenproblem(self):
        """
        Solve eigenvalue problem and perform mass normalization.

        Returns
        -------
        frequencies : np.ndarray
            Natural frequencies (Hz).
        eigenvectors : np.ndarray
            Mass-normalized mode shapes.
        """
        eigenvalues, eigenvectors = eigh(self.K_global, self.M_global)
        frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

        # Mass-normalize eigenvectors
        norms = np.sqrt(np.einsum('ij,ij->j', eigenvectors.T @ self.M_global, eigenvectors.T))
        eigenvectors /= norms

        return frequencies, eigenvectors

    def extract_submatrices(self, global_matrix):
        """
        Extract displacement and rotation submatrices.

        Parameters
        ----------
        global_matrix : np.ndarray
            Global matrix to split.

        Returns
        -------
        K_uu, K_uθ, K_θu, K_θθ : np.ndarray
            Submatrices for displacement and rotation DOFs.
        """
        K_uu = global_matrix[::2, ::2]
        K_uθ = global_matrix[::2, 1::2]
        K_θu = global_matrix[1::2, ::2]
        K_θθ = global_matrix[1::2, 1::2]
        return K_uu, K_uθ, K_θu, K_θθ

    def split_eigenvectors(self, eigenvectors):
        """
        Split eigenvectors into displacement and rotation components.

        Parameters
        ----------
        eigenvectors : np.ndarray
            Eigenvector matrix.

        Returns
        -------
        disp, rot : np.ndarray
            Displacement and rotation components.
        """
        return eigenvectors[::2], eigenvectors[1::2]

def plot_modes(node_coords, eigenvectors, mode_indices):
    """
    Plot selected mode shapes.

    Parameters
    ----------
    node_coords : np.ndarray
        Coordinates of beam nodes.
    eigenvectors : np.ndarray
        Mode shapes.
    mode_indices : list
        Indices of modes to plot.
    """
    plt.figure(figsize=(8, 6))
    for idx, mode_index in enumerate(mode_indices):
        plt.plot(node_coords, eigenvectors[:, mode_index], label=f'Mode {mode_index + 1}')
    plt.xlabel('Length (m)')
    plt.ylabel('Amplitude')
    plt.title('Mode Shapes')
    plt.legend()
    plt.grid()
    plt.show()

def normalize_by_max(eigenvectors):
    """
    Normalize mode shapes by maximum absolute value.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Mode shapes.

    Returns
    -------
    np.ndarray
        Normalized mode shapes.
    """
    return eigenvectors / np.max(np.abs(eigenvectors), axis=0)
