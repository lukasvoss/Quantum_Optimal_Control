import numpy as np

from braket.circuits import Circuit
from braket.circuits.circuit import Instruction
from braket.circuits.gates import U, PhaseShift, Ry


def transpile_U(circuit: Circuit):

    def decompose_U_rotations(instructions, angles: tuple, target: int):
        instructions.append(Instruction(PhaseShift(angle=angles[0]), target))
        instructions.append(Instruction(Ry(angle=angles[1]), target))
        instructions.append(Instruction(PhaseShift(angle=angles[2]), target))
        return instructions

    instructions = circuit.instructions
    compatible_instructions = []
    for instr in instructions:        
        if isinstance(instr.operator, U):
            target = instr.target
            angles = (instr.operator.angle_1, instr.operator.angle_2, instr.operator.angle_3)
            decompose_U_rotations(compatible_instructions, angles, target)
        else:
            compatible_instructions.append(instr)
        
    return Circuit(compatible_instructions)