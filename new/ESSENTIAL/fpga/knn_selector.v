// ============================================================================
// Optimized K-Nearest Neighbors Selector
// Instead of full sort, maintains only K smallest distances efficiently.
// Uses parallel comparators to find k nearest neighbors in fewer cycles.
// ============================================================================

module knn_selector #(
    parameter DATA_WIDTH       = 8,
    parameter NUM_TRAINING     = 50,
    parameter K_VALUE          = 3,
    parameter DIST_WIDTH       = 12
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              start,
    input  wire [DIST_WIDTH-1:0]             distances [0:NUM_TRAINING-1],
    input  wire                              distances_valid,
    
    output reg  [7:0]                        selected_indices [0:K_VALUE-1],
    output reg  [DIST_WIDTH-1:0]             selected_distances [0:K_VALUE-1],
    output reg                               selection_valid,
    output reg  [7:0]                        progress
);

    // ── State Encoding ──────────────────────────────────────────────────
    localparam IDLE       = 2'd0;
    localparam SELECTING  = 2'd1;
    localparam DONE       = 2'd2;

    reg [1:0] state, next_state;
    reg [7:0] sample_idx;
    
    // ── K-Selection Registers ───────────────────────────────────────────
    // Keep track of K smallest distances and their original indices
    reg [DIST_WIDTH-1:0] k_distances [0:K_VALUE-1];
    reg [7:0]            k_indices [0:K_VALUE-1];
    reg [7:0]            k_count;  // Number of elements in k-pool
    
    // ── Temporary registers for finding max in k-pool ──────────────────
    reg [DIST_WIDTH-1:0] max_in_pool;
    reg [7:0]            max_idx_in_pool;
    integer i;
    
    // ── Find maximum distance in current k-pool (combinational) ─────────
    always @(*) begin
        max_in_pool = k_distances[0];
        max_idx_in_pool = 0;
        for (i = 1; i < K_VALUE; i = i + 1) begin
            if (k_distances[i] > max_in_pool) begin
                max_in_pool = k_distances[i];
                max_idx_in_pool = i;
            end
        end
    end
    
    // ── Sequential Logic ────────────────────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            sample_idx <= 0;
            k_count <= 0;
            progress <= 0;
            selection_valid <= 0;
            
            for (i = 0; i < K_VALUE; i = i + 1) begin
                k_distances[i] <= {DIST_WIDTH{1'b1}};  // Initialize to max
                k_indices[i] <= 0;
            end
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    sample_idx <= 0;
                    k_count <= 0;
                    progress <= 0;
                    selection_valid <= 0;
                    
                    for (i = 0; i < K_VALUE; i = i + 1) begin
                        k_distances[i] <= {DIST_WIDTH{1'b1}};
                        k_indices[i] <= 0;
                    end
                end
                
                SELECTING: begin
                    if (distances_valid && sample_idx < NUM_TRAINING) begin
                        if (k_count < K_VALUE) begin
                            // Pool not full yet - just add
                            k_distances[k_count] <= distances[sample_idx];
                            k_indices[k_count] <= sample_idx;
                            k_count <= k_count + 1;
                        end else if (distances[sample_idx] < max_in_pool) begin
                            // Replace max in pool with current distance
                            k_distances[max_idx_in_pool] <= distances[sample_idx];
                            k_indices[max_idx_in_pool] <= sample_idx;
                        end
                        
                        sample_idx <= sample_idx + 1;
                        progress <= (sample_idx * 8'd255) / NUM_TRAINING;
                    end
                end
                
                DONE: begin
                    selection_valid <= 1;
                    progress <= 8'd255;
                    
                    // Output k-nearest in order
                    for (i = 0; i < K_VALUE; i = i + 1) begin
                        selected_distances[i] <= k_distances[i];
                        selected_indices[i] <= k_indices[i];
                    end
                end
            endcase
        end
    end
    
    // ── Combinational Next-State Logic ──────────────────────────────────
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:     if (start && distances_valid)    next_state = SELECTING;
            SELECTING: if (sample_idx >= NUM_TRAINING) next_state = DONE;
            DONE:      if (!start)                     next_state = IDLE;
            default:   next_state = IDLE;
        endcase
    end

endmodule
