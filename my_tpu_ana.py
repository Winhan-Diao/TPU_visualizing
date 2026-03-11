from typing import List, Tuple, Optional, Dict, Union
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
from datetime import datetime
import random

mat_usage_per_clk = []
vpu_usage_per_clk = []
global_clk = 0
detailed_per_clk_tmp = []
detailed_per_clk = pd.DataFrame([])
x_in_per_clk: List[List[List[Optional[float]]]] = []
w_in_per_clk: List[List[List[Optional[float]]]] = []

def simplify_cast(x: Optional[float]):
    if x is None:
        return 2
    if x:
        return 1
    return 0

class pe():
    # weight to forward to the next PE
    reg_pass: float
    # weight used to multiple by X
    reg_time: float
    # <Not Necessary> x to pass to the sibling PE
    # reg_x: Optional[float]
    def __init__(self):
        self.reg_pass = 0.
        self.reg_time = 0.
        # self.reg_x = None
    def clk(self, x: Optional[Tuple[float, int]], weight: Optional[float], psum: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        out_weight = None
        out_psum = None
        out_x = x
        # self.reg_x = x
        psum = 0. if psum is None else psum
        if weight is not None:
            out_weight = self.reg_pass
            self.reg_pass = weight
        if x is not None:
            self.reg_time = self.reg_pass
            out_psum = self.reg_time * x[0] + psum
        return out_weight, out_x, out_psum

class systolic_matrix():
    n_length: int
    pes: List[List[pe]]
    wxps: List[List[Tuple[Optional[float], Optional[Tuple[float, int]], Optional[float]]]]
    x_in: List[List[Optional[Tuple[float, int]]]]
    w_in: List[List[Optional[float]]]
    def __init__(self, n_length = 2):
        self.n_length = n_length
        # Create n_length x n_length matrix of PEs
        self.pes = [[pe() for _ in range(n_length)] for _ in range(n_length)]
        # Initialize wxps with None values
        self.wxps = [[(None, None, None) for _ in range(n_length)] for _ in range(n_length)]
        # Initialize input queues
        self.x_in = [[] for _ in range(n_length)]
        self.w_in = [[] for _ in range(n_length)]
    
    def clk(self) -> List[Optional[float]]:
        x_in_record = [[simplify_cast(x[0] if x is not None else None) for x in l] for l in self.x_in]
        x_in_per_clk.append(x_in_record)
        w_in_record = [[simplify_cast(w) for w in l] for l in self.w_in]
        w_in_per_clk.append(w_in_record)
        mat_usage = 0
        prev_wxps = deepcopy(self.wxps)
        for i in range(1, self.n_length):
            if not len(self.x_in[i]) and prev_wxps[i-1][0][1] is not None:
                self.x_in[i].append((0., prev_wxps[i-1][0][1][1]))
        pass
        # Process each PE in the systolic array
        for i in range(self.n_length):
            for j in range(self.n_length):
                # Get inputs for this PE
                x_input = None
                weight_input = None
                psum_input = 0.
                # Handle x input (from left or from input queue)
                if j == 0:
                    # Leftmost column gets x from input queue
                    x_input = self.x_in[i].pop(0) if len(self.x_in[i]) else None
                else:
                    # Other columns get x from the left neighbor
                    x_input = prev_wxps[i][j-1][1] if prev_wxps[i][j-1][1] is not None and j < prev_wxps[i][j-1][1][1] else None
                # Handle weight input (from top or from input queue)
                if i == 0:
                    # Top row gets weight from input queue
                    weight_input = self.w_in[j].pop(0) if len(self.w_in[j]) else None
                else:
                    # Other rows get weight from top neighbor
                    weight_input = prev_wxps[i-1][j][0]
                # Handle partial sum input (from top neighbor)
                if i == 0:
                    # Top row starts with 0 partial sum
                    psum_input = 0.
                else:
                    # Other rows get partial sum from top neighbor
                    psum_input = prev_wxps[i-1][j][2]
                # Clock the PE
                self.wxps[i][j] = self.pes[i][j].clk(x_input, weight_input if i == 0 else self.wxps[i-1][j][0], psum_input)
                if self.wxps[i][j][2] is not None:
                    mat_usage += 1
                detailed_per_clk_tmp.append((global_clk, f"pe_{i}_{j}_wire_w", simplify_cast(self.wxps[i][j][0])))
                detailed_per_clk_tmp.append((global_clk, f"pe_{i}_{j}_wire_x", simplify_cast(self.wxps[i][j][1][0] if self.wxps[i][j][1] is not None else None)))
                detailed_per_clk_tmp.append((global_clk, f"pe_{i}_{j}_wire_p", simplify_cast(self.wxps[i][j][2])))
                detailed_per_clk_tmp.append((global_clk, f"pe_{i}_{j}_reg_w_used", simplify_cast(self.pes[i][j].reg_time)))
                detailed_per_clk_tmp.append((global_clk, f"pe_{i}_{j}_reg_w_pass", simplify_cast(self.pes[i][j].reg_pass)))
        mat_usage /= self.n_length * self.n_length
        mat_usage_per_clk.append(mat_usage)
        # Return the partial sums from the bottom row
        return [self.wxps[self.n_length-1][j][2] for j in range(self.n_length)]

class vpu_pipeline():
    reg_bias: Optional[float]
    reg_leakyrelu: Optional[float]
    reg_mse: Optional[float]
    reg_d_leakyrelu: Optional[float]
    b_in: List[float]
    def __init__(self):
        self.reg_bias = None
        self.reg_leakyrelu = None
        self.reg_mse = None
        self.reg_d_leakyrelu = None
        self.b_in = []
        self.vpu_pipeline_usage = 0
    @staticmethod
    def clk_bias(input: Optional[float], bias: Optional[float]):
        if input is None or bias is None:
            return None
        return input + bias
    @staticmethod
    def clk_leakyrelu(input: Optional[float]):
        if input is None:
            return None
        return .50 * input if input < 0 else input
    @staticmethod
    def clk_maybe_mse(maybe_pred: Optional[float], actual: Optional[float]):
        if actual is None and maybe_pred is not None:
            return maybe_pred
        elif actual is not None and maybe_pred is not None:
            return .5 * (maybe_pred - actual)
        return None
    @staticmethod
    def clk_maybe_d_leakyrelu(maybe_loss: Optional[float], pred: Optional[float]):
        if pred is None and maybe_loss is not None:
            return maybe_loss
        elif maybe_loss is not None and pred is not None:
            return .50 * maybe_loss if pred < 0 else maybe_loss
        return None
    def clk(self, input: Optional[float], actual: Optional[float], pred: Optional[float]) -> Optional[float]:
        self.vpu_pipeline_usage = 0
        prev_reg_bias = self.reg_bias
        prev_reg_leakyrelu = self.reg_leakyrelu
        prev_reg_mse = self.reg_mse
        prev_reg_d_leakyrelu = self.reg_d_leakyrelu
        self.reg_bias = self.clk_bias(input, None if input is None or not len(self.b_in) else self.b_in.pop(0))
        self.reg_leakyrelu = self.clk_leakyrelu(prev_reg_bias)
        self.reg_mse = self.clk_maybe_mse(prev_reg_leakyrelu, actual)
        self.reg_d_leakyrelu = self.clk_maybe_d_leakyrelu(prev_reg_mse, pred)
        if self.reg_bias is not None:
            self.vpu_pipeline_usage += 1
        if self.reg_leakyrelu is not None:
            self.vpu_pipeline_usage += 1
        if self.reg_mse is not None:
            self.vpu_pipeline_usage += 1
        if self.reg_d_leakyrelu is not None:
            self.vpu_pipeline_usage += 1
        self.vpu_pipeline_usage /= 4
        return prev_reg_d_leakyrelu

class vpu():
    pipelines: List[vpu_pipeline]
    def __init__(self, n_length=2):
        self.pipelines = [vpu_pipeline() for _ in range(n_length)]
    
    def clk(self, inputs: List[Optional[float]], ) -> List[Optional[float]]:
        outputs = []
        for i, pipeline in enumerate(self.pipelines):
            outputs.append(pipeline.clk(inputs[i] if i < len(inputs) else None, None, None))
            detailed_per_clk_tmp.append((global_clk, f"vpu_{i}_reg_bias", simplify_cast(pipeline.reg_bias)))
            detailed_per_clk_tmp.append((global_clk, f"vpu_{i}_reg_leakyrelu", simplify_cast(pipeline.reg_leakyrelu)))
            detailed_per_clk_tmp.append((global_clk, f"vpu_{i}_reg_mse", simplify_cast(pipeline.reg_mse)))
            detailed_per_clk_tmp.append((global_clk, f"vpu_{i}_reg_d_leakyrelu", simplify_cast(pipeline.reg_d_leakyrelu)))
        vpu_usage_per_clk.append(sum(p.vpu_pipeline_usage for p in self.pipelines) / len(self.pipelines))
        return outputs

class tpu():
    vpu_: vpu
    mat: systolic_matrix
    mat_len: int
    # reg_mat_outputs: List[Optional[float]]
    def __init__(self, n_length=2):
        self.mat_len = n_length
        self.vpu_ = vpu(n_length)
        self.mat = systolic_matrix(n_length)
        self.reg_mat_outputs = [None] * n_length
        
    def clk(self):
        global global_clk
        global_clk += 1
        reg_mat_outputs = self.mat.clk()
        vpu_outputs = self.vpu_.clk(reg_mat_outputs)
        # print(reg_mat_outputs, "->", vpu_outputs)       # outputs
        return vpu_outputs
    
    def forward(self, n_layers: int, weights: List[List[List[float]]], biases: List[List[float]], xs: List[List[float]]):
        # Append biases in advance because it is hard to calculate the precise used time
        for layer in range(n_layers):
            for i, pipeline in enumerate(self.vpu_.pipelines):
                pipeline.b_in.extend([biases[layer][i] if i < len(biases[layer]) else None] * len(xs))
        overlapped_clks = 0
        xs_in_unfinished = [0] * len(xs[0])
        xs_in_unfinished_next = [len(xs)] * self.mat_len
        xs_buffer = [[] for _ in range(self.mat_len)]
        for layer in range(n_layers):
            # print("layer:", layer)        # outputs
            if layer == 0:
                # Process inputs through the systolic array
                for batch in range(len(xs)):
                    # for _ in range(self.mat_len - len(xs[batch])):
                    #     xs[batch].append(0.)
                    if batch == 0:
                        for from_x in range(len(xs[batch])):
                            # Add padding for weights to come in first
                            for _ in range(1 + from_x):
                                self.mat.x_in[from_x].append(None)
                    # Add actual input values
                    for from_x in range(len(xs[batch])):
                        self.mat.x_in[from_x].append((xs[batch][from_x], len(weights[layer])))
                # Load weights for this layer
                for to_node in range(len(weights[layer])):
                    for _ in range(to_node):
                        self.mat.w_in[to_node].append(None)
                    for from_node in range(len(weights[layer][to_node])):
                        self.mat.w_in[to_node].append(weights[layer][to_node][from_node])            
                # Process through clock cycles
            # Calculate required clock cycles: input_size + output_size - 1 + additional cycles
            total_cycles = len(xs) + self.mat_len - 1 + len(weights[layer]) + 4 - overlapped_clks
            overlapped_clks = 0
            already_overlapped = False
            xs_in_unfinished_next_next = [len(xs)] * self.mat_len
            for t in range(total_cycles):
                vpu_outputs = self.clk()
                for to_x in range(len(weights[layer])):
                    if vpu_outputs[to_x] is not None:
                        if xs_in_unfinished[to_x]:
                            xs_in_unfinished[to_x] -= 1
                            self.mat.x_in[to_x].append((vpu_outputs[to_x], len(weights[layer])))
                        elif not xs_in_unfinished_next[to_x]:
                            if layer == n_layers - 1:
                                continue
                            if xs_in_unfinished_next[to_x]:
                                xs_in_unfinished_next[to_x] -= 1
                            else:
                                xs_in_unfinished_next_next[to_x] -= 1
                            xs_buffer[to_x].append((vpu_outputs[to_x], len(weights[layer + 1])))
                        else:
                            if layer == n_layers - 1:
                                continue
                            if xs_in_unfinished_next[to_x]:
                                xs_in_unfinished_next[to_x] -= 1
                            else:
                                xs_in_unfinished_next_next[to_x] -= 1
                            if not already_overlapped:
                                already_overlapped = True
                                overlapped_clks = total_cycles - t - 1 - len(self.mat.x_in[0])
                                for to_node in range(len(weights[layer + 1])):
                                    for _ in range(to_node + len(self.mat.x_in[0])):
                                        self.mat.w_in[to_node].append(None)
                                    for from_node in range(len(weights[layer + 1][to_node])):
                                        self.mat.w_in[to_node].append(weights[layer + 1][to_node][from_node])            
                                var0 = len(self.mat.x_in[0])
                                for from_x in range(self.mat_len):
                                    if from_x and not len(self.mat.x_in[from_x]) and self.mat.wxps[from_x][0][1] is not None:
                                        self.mat.x_in[from_x].extend([(0., len(weights[layer + 1]))] * (from_x - len(self.mat.x_in[from_x]) + var0))
                                        self.mat.x_in[from_x].append(None)
                                    else:
                                        self.mat.x_in[from_x].extend([None] * (from_x + 1 - len(self.mat.x_in[from_x]) + var0))
                                for from_x in range(len(xs_buffer)):
                                    self.mat.x_in[from_x].extend(xs_buffer[from_x])
                                xs_buffer = [[] for _ in range(len(weights[layer]))]
                            if vpu_outputs[to_x] is not None:
                                self.mat.x_in[to_x].append((vpu_outputs[to_x], len(weights[layer + 1])))
            xs_in_unfinished = xs_in_unfinished_next
            xs_in_unfinished_next = xs_in_unfinished_next_next
            # Transpose the results to get the new inputs for next layer
            # xs = [list(row) for row in zip(*xs_new_t)])

def get_usage_data():
    """Returns the usage data for MAT and VPU in a JSON serializable format"""
    file_name_prefix = f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{random.randint(0, 99999_99999):010d}"
    detailed_per_clk_file_name = f"tmp/{file_name_prefix}_details.json"
    detailed_per_clk_tmp_2 = pd.DataFrame(detailed_per_clk_tmp, columns=["clk", "type", "state"])
    detailed_per_clk = detailed_per_clk_tmp_2.pivot_table(index="clk", columns=["type"], values="state").astype(np.int8)
    detailed_per_clk.to_json("./public/" + detailed_per_clk_file_name)
    x_in_per_clk_file_name = f"tmp/{file_name_prefix}_x_in.json"
    w_in_per_clk_file_name = f"tmp/{file_name_prefix}_w_in.json"
    with open("./public/" + x_in_per_clk_file_name, "w", encoding="utf-8") as f:
        json.dump(x_in_per_clk, f, ensure_ascii=False, indent=2)
    with open("./public/" + w_in_per_clk_file_name, "w", encoding="utf-8") as f:
        json.dump(w_in_per_clk, f, ensure_ascii=False, indent=2)
    return {
        "mat_usage": mat_usage_per_clk,
        "vpu_usage": vpu_usage_per_clk,
        "timestamps": list(range(len(mat_usage_per_clk))),  # Assuming each clock cycle is 1 time unit
        "details": detailed_per_clk_file_name,
        "x_in": x_in_per_clk_file_name,
        "w_in": w_in_per_clk_file_name
    }

def reset_usage_data():
    """Resets the usage data lists"""
    global mat_usage_per_clk, vpu_usage_per_clk, global_clk
    mat_usage_per_clk = []
    vpu_usage_per_clk = []
    global_clk = 0

def run_simulation(n_layers=24, n_nodes=3, batch_size=32, n_length=3):
    """Runs a simulation with given parameters"""
    reset_usage_data()  # Clear previous data
    tpu_ = tpu(n_length=n_length)
    tpu_.forward(
        n_layers, 
        np.random.randint(1, 2, size=(n_layers, n_nodes, n_nodes)).tolist(), 
        np.random.randint(1, 2, size=(n_layers, n_nodes)).tolist(), 
        np.random.randint(1, 2, size=(batch_size, n_nodes)).tolist()
    )
    return get_usage_data()

# All unit can only step once per clk.
if __name__ == "__main__":
    # run xor once forwards and backwards    
    # tpu_ = tpu(n_length=10)
    # tpu_.forward(2, 
    #     [
    #         [[-.579, .299], [.423, .091]], 
    #         [[.296, .527], ]
    #     ], 
    #     [
    #         [-.494, .189], 
    #         [.636, ]
    #     ],
    #     [
    #         [0., 0.], 
    #         [0., 1.], 
    #         [1., 0.], 
    #         [1., 1.],
    #     ]
    # )
    n_layers = 9
    n_nodes = 10
    batch_size = 16
    n_length = 10
    tpu_ = tpu(n_length=n_length)
    tpu_.forward(n_layers, np.random.randint(1, 2, size=(n_layers, n_nodes, n_nodes)).tolist(), np.random.randint(1, 2, size=(n_layers, n_nodes)).tolist(), np.random.randint(1, 2, size=(batch_size, n_nodes)).tolist())
    print(mat_usage_per_clk)
    print(vpu_usage_per_clk)
    detailed_per_clk_tmp_2 = pd.DataFrame(detailed_per_clk_tmp, columns=["clk", "type", "state"])
    detailed_per_clk = detailed_per_clk_tmp_2.pivot_table(index="clk", columns=["type"], values="state")
    detailed_per_clk_json = detailed_per_clk.to_json("./tmp.json", indent=2)
    fig = plt.figure(dpi=200, figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.plot(mat_usage_per_clk)[0].set_label("MAT")
    ax.plot(vpu_usage_per_clk)[0].set_label("VPU")
    plt.show()