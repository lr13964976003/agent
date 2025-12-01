#!/usr/bin/env python3

def fix_gpu_mapping_dag_comprehensive():
    """Fix the GPU mapping DAG by removing all cycles"""
    input_file = "../outputs/2025-12-01-14-11-15/gpu_mapping_dag.dot"
    output_file = "../outputs/2025-12-01-14-11-15/gpu_mapping_dag_final.dot"
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    fixed_lines = []
    
    # Track which edges we've seen to avoid cycles
    expert_a2a_edges_seen = set()
    tp_edges_seen = set()
    
    for line in lines:
        if '->' in line:
            # Parse the edge
            parts = line.split('->')
            if len(parts) == 2:
                src = parts[0].strip()
                dst_part = parts[1].strip()
                dst = dst_part.split('[')[0].strip()
                
                # Handle Expert All-to-All edges
                if 'Expert A2A' in line:
                    # For expert A2A, only keep edges from lower to higher GPU numbers
                    # This creates a directed acyclic pattern
                    src_num = int(src.replace('gpu_', ''))
                    dst_num = int(dst.replace('gpu_', ''))
                    
                    if src_num < dst_num:  # Only keep forward edges
                        fixed_lines.append(line)
                
                # Handle Tensor Parallelism edges
                elif 'TP All-Reduce' in line:
                    # For TP, only keep one direction per pair
                    # Create a canonical edge representation (smaller node first)
                    edge = tuple(sorted([src, dst]))
                    
                    if edge not in tp_edges_seen:
                        tp_edges_seen.add(edge)
                        # Always go from even to odd GPU within the same pipeline stage
                        src_num = int(src.replace('gpu_', ''))
                        dst_num = int(dst.replace('gpu_', ''))
                        
                        # TP pairs are (0,1), (2,3), (4,5), etc.
                        # Keep only the direction from even to odd
                        if src_num % 2 == 0 and dst_num == src_num + 1:
                            fixed_lines.append(line)
                
                # Handle Pipeline edges (these should be fine as they're already directed)
                elif 'Pipeline Send' in line:
                    fixed_lines.append(line)
                
                else:
                    # Keep any other edges as-is
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write the fixed content
    fixed_content = '\n'.join(fixed_lines)
    
    with open(output_file, 'w') as f:
        f.write(fixed_content)
    
    print(f"Comprehensively fixed GPU mapping DAG saved to {output_file}")
    return output_file

if __name__ == "__main__":
    fix_gpu_mapping_dag_comprehensive()