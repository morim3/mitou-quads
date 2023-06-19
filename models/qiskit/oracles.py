from qiskit import QuantumCircuit
from qiskit.extensions import UnitaryGate
import numpy as np



class AbsoluteValue(QuantumCircuit):
    """
    絶対値関数|x|の量子回路. (|x>|0> => |x>| | |x| >)
    """
    def __init__(self, n_digits: int):
        circuit = QuantumCircuit(2 * n_digits, name="abs")
        dist_bits = range(n_digits)
        func_bits = range(n_digits, 2 * n_digits)
        for i in range(len(func_bits) - 1):
            circuit.ccx(dist_bits[i], dist_bits[-1],  func_bits[i])
            circuit.x(dist_bits[-1])
            circuit.x(dist_bits[i])
            circuit.ccx(dist_bits[i], dist_bits[-1],  func_bits[i])
            circuit.x(dist_bits[i])
            circuit.x(dist_bits[-1])
        
        super().__init__(*circuit.qregs, name="abs")
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

def diagonal_oracle(func, threshold, n_digits: int):
    test_point = np.arange(2**n_digits)
    test_val = np.where(func(test_point / 2 ** n_digits) < threshold, -1, 1)
    mat = np.diag(test_val)
    return UnitaryGate(mat)


class OneDimFuncOracle(QuantumCircuit):
    """
    定義域[0, 1]の一次元関数について, しきい値以下の領域をflipさせる量子回路.
    """
    def __init__(self, func, threshold, n_digits: int):

        circuit = QuantumCircuit(n_digits, name="func_oracle")
        flagged_regions = self.get_flagged_regions(func, threshold, n_digits)

        for begin, end in flagged_regions:
            begin = format(begin, f"0{n_digits}b")
            end = format(end, f"0{n_digits}b")

            start_ind = 0
            while start_ind < n_digits and begin[start_ind] == end[start_ind]:
                start_ind += 1

            for ind in range(start_ind, n_digits):
                if begin[ind] == '0':
                    self.add_control_z(circuit, begin[0:ind-1]+'1', n_digits)

                if ind == n_digits - 1:
                    self.add_control_z(circuit, begin, n_digits)
                                                  
            for ind in range(start_ind, n_digits):
                if end[ind] == '1':
                    self.add_control_z(circuit, end[0:ind-1]+'0', n_digits)

                if ind == n_digits - 1:
                    self.add_control_z(circuit, end, n_digits)


        super().__init__(*circuit.qregs, name="func_oracle")
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

    @staticmethod
    def add_control_z(circuit, register_str: str, n_digits: int):
        for i, s in enumerate(register_str):
            if int(s) == 0:
                circuit.x(n_digits - i - 1)

        circuit.z(n_digits-1)
        if len(register_str) > 1:
            circuit.mcx(list(range(n_digits-len(register_str), n_digits-1)), n_digits-1)
        else:
            circuit.x(n_digits-1)
        circuit.z(n_digits-1)

        for i, s in enumerate(register_str):
           if int(s) == 0:
               circuit.x(n_digits - i - 1)        

    @staticmethod
    def get_flagged_regions(func, threshold, n_digits):
        test_point = np.arange(2**n_digits)
        test_val = func(test_point / 2 ** n_digits) < threshold
        flagged_regions = []
         
        region_begin = -1
        for p, v in zip(test_point, test_val):
            
            if v == False:
                if region_begin == -1:
                    continue
                else:
                    region_end = p - 1
                    flagged_regions.append((region_begin, region_end))
                    region_begin = -1
            else:
                if region_begin == -1:
                    region_begin = p
                else:
                    continue

        if region_begin != -1:
            flagged_regions.append((region_begin, 2**n_digits-1))
        
        return
