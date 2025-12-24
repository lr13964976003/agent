#!/usr/bin/env python3

import re

def fix_dag_connections():
    """Fix remaining connection issues in the DAG"""
    
    # Read the current DOT file
    with open("../outputs/2025-12-24-15-21-58/moe_complete_dag.dot", "r") as f:
        content = f.read()
    
    # Add missing connections for nodes with only in-degree:
    # - dec_gpu1_l0_gate (needs input from dec_gpu1_l0_ln1)
    # - decode_output_agg (needs proper input connections)
    # - final_output (needs proper input connections)
    
    # Add missing connections for nodes with only out-degree:
    # - input (should connect to decode_input as well)
    # - decode_input (needs to connect to all decode phases)
    
    # Find the section with decode connections and fix them
    decode_section = """    // Decode connections
    decode_input -> dec_gpu0_l0_ln1;
    decode_input -> dec_gpu1_l0_ln1;
    decode_input -> dec_gpu2_recv;
    decode_input -> dec_gpu3_recv;
    
    dec_gpu0_l0_ln1 -> dec_gpu0_l0_attn_q;
    dec_gpu0_l0_attn_q -> dec_gpu0_l0_attn_ar;
    dec_gpu0_l0_attn_ar -> dec_gpu0_l0_ln2;
    dec_gpu0_l0_ln2 -> dec_gpu0_send;
    
    dec_gpu1_l0_ln1 -> dec_gpu1_l0_gate;
    dec_gpu1_l0_ln1 -> dec_gpu1_ep_a2a_send;
    dec_gpu1_ep_a2a_send -> dec_gpu1_ep_a2a_recv;
    dec_gpu1_ep_a2a_recv -> dec_gpu1_send;
    
    dec_gpu0_send -> dec_gpu2_recv;
    dec_gpu1_send -> dec_gpu3_recv;
    
    dec_gpu2_recv -> dec_gpu2_l8_ln1;
    dec_gpu2_l8_ln1 -> dec_gpu2_l8_attn_q;
    dec_gpu2_l8_attn_q -> dec_gpu2_l8_attn_ar;
    dec_gpu2_l8_attn_ar -> dec_gpu2_output;
    
    dec_gpu3_recv -> dec_gpu3_l8_ln1;
    dec_gpu3_l8_ln1 -> dec_gpu3_l8_attn_q;
    dec_gpu3_l8_attn_q -> dec_gpu3_l8_attn_ar;
    dec_gpu3_l8_attn_ar -> dec_gpu3_output;
    
    dec_gpu2_output -> decode_output_agg;
    dec_gpu3_output -> decode_output_agg;"""
    
    # Replace with corrected decode section
    corrected_decode_section = """    // Decode connections
    decode_input -> dec_gpu0_l0_ln1;
    decode_input -> dec_gpu1_l0_ln1;
    decode_input -> dec_gpu2_recv;
    decode_input -> dec_gpu3_recv;
    
    dec_gpu0_l0_ln1 -> dec_gpu0_l0_attn_q;
    dec_gpu0_l0_attn_q -> dec_gpu0_l0_attn_ar;
    dec_gpu0_l0_attn_ar -> dec_gpu0_l0_ln2;
    dec_gpu0_l0_ln2 -> dec_gpu0_send;
    
    dec_gpu1_l0_ln1 -> dec_gpu1_l0_gate;
    dec_gpu1_l0_gate -> dec_gpu1_ep_a2a_send;  // Fixed: gate needs to control routing
    dec_gpu1_l0_ln1 -> dec_gpu1_ep_a2a_send;
    dec_gpu1_ep_a2a_send -> dec_gpu1_ep_a2a_recv;
    dec_gpu1_ep_a2a_recv -> dec_gpu1_send;
    
    dec_gpu0_send -> dec_gpu2_recv;
    dec_gpu1_send -> dec_gpu3_recv;
    
    dec_gpu2_recv -> dec_gpu2_l8_ln1;
    dec_gpu2_l8_ln1 -> dec_gpu2_l8_attn_q;
    dec_gpu2_l8_attn_q -> dec_gpu2_l8_attn_ar;
    dec_gpu2_l8_attn_ar -> dec_gpu2_output;
    
    dec_gpu3_recv -> dec_gpu3_l8_ln1;
    dec_gpu3_l8_ln1 -> dec_gpu3_l8_attn_q;
    dec_gpu3_l8_attn_q -> dec_gpu3_l8_attn_ar;
    dec_gpu3_l8_attn_ar -> dec_gpu3_output;
    
    dec_gpu2_output -> decode_output_agg;
    dec_gpu3_output -> decode_output_agg;
    
    // Add missing output connections
    decode_output_agg -> final_output;  // Connect decode output to final output
    input -> decode_input;  // Connect input to decode input for completeness"""
    
    # Replace the decode section
    content = content.replace(decode_section, corrected_decode_section)
    
    # Write the corrected content
    with open("../outputs/2025-12-24-15-21-58/moe_complete_dag_fixed.dot", "w") as f:
        f.write(content)
    
    print("Fixed DAG connections written to moe_complete_dag_fixed.dot")
    return "../outputs/2025-12-24-15-21-58/moe_complete_dag_fixed.dot"

if __name__ == "__main__":
    fixed_file = fix_dag_connections()
    print(f"Fixed file: {fixed_file}")