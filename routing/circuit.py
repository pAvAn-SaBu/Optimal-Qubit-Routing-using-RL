"""Circuits as an ordered list of two-qubit gates.

Single-qubit gates never need routing (they run wherever the qubit sits), so a
circuit here is only its two-qubit gates, in program order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    gates: tuple[tuple[int, int], ...]  # (control, target) logical qubits

    def __post_init__(self) -> None:
        for a, b in self.gates:
            if a == b or not (0 <= a < self.n_qubits and 0 <= b < self.n_qubits):
                raise ValueError(f"bad gate ({a}, {b}) for {self.n_qubits} qubits")

    def __len__(self) -> int:
        return len(self.gates)

    def to_qiskit(self, n_physical: int | None = None) -> QuantumCircuit:
        qc = QuantumCircuit(n_physical or self.n_qubits)
        for a, b in self.gates:
            qc.cx(a, b)
        return qc

    @classmethod
    def from_qiskit(cls, qc: QuantumCircuit) -> "Circuit":
        gates = []
        dropped = 0
        for inst in qc.data:
            qubits = [qc.find_bit(q).index for q in inst.qubits]
            if len(qubits) == 2:
                gates.append((qubits[0], qubits[1]))
            elif len(qubits) > 2:
                raise ValueError(f"gate {inst.operation.name} acts on {len(qubits)} qubits; decompose it first")
            else:
                dropped += 1
        if dropped:
            log.info("ignored %d single-qubit gates/ops (they never need routing)", dropped)
        return cls(qc.num_qubits, tuple(gates))

    @classmethod
    def parse(cls, text: str, n_qubits: int) -> "Circuit":
        """Parse '0-3, 1-2, 0-1' into gates CX(0,3), CX(1,2), CX(0,1)."""
        gates = []
        for tok in text.replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                a, b = (int(x) for x in tok.split("-"))
                gates.append((a, b))
        return cls(n_qubits, tuple(gates))


def random_circuit(n_qubits: int, n_gates: int, rng: np.random.Generator) -> Circuit:
    gates = []
    for _ in range(n_gates):
        a, b = rng.choice(n_qubits, size=2, replace=False)
        gates.append((int(a), int(b)))
    return Circuit(n_qubits, tuple(gates))
