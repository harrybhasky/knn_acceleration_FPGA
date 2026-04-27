// ============================================================================
// Parameterized Distance Metric Module
// Calculates distance between test features and all training samples.
// Supports both Manhattan (L1) and Euclidean (L2) distance metrics.
//
// DISTANCE_METRIC = 0 : Manhattan (L1) distance = |x1-x2| + |y1-y2|
// DISTANCE_METRIC = 1 : Euclidean (L2) squared distance = (x1-x2)^2 + (y1-y2)^2
//                       (We avoid sqrt to keep computation efficient)
// ============================================================================

module distance_metric #(
    parameter DATA_WIDTH       = 8,
    parameter NUM_FEATURES     = 2,
    parameter NUM_TRAINING     = 50,
    parameter DISTANCE_METRIC  = 0,     // 0=Manhattan, 1=Euclidean
    parameter DIST_WIDTH       = 12     // Width for distance values
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              start,
    input  wire [DATA_WIDTH*NUM_FEATURES-1:0] test_features,
    output reg  [DIST_WIDTH-1:0]             distances [0:NUM_TRAINING-1],
    output reg                               distances_valid,
    output reg  [7:0]                        progress  // Debug: calculation progress
);

    // ── State Encoding ──────────────────────────────────────────────────
    localparam IDLE       = 2'd0;
    localparam CALCULATING = 2'd1;
    localparam DONE       = 2'd2;

    reg [1:0] state, next_state;
    reg [7:0] sample_idx;
    
    // ── Extract Input Features (assumes NUM_FEATURES=2) ──────────────────
    wire [DATA_WIDTH-1:0] test_f0 = test_features[DATA_WIDTH*NUM_FEATURES-1:DATA_WIDTH];
    wire [DATA_WIDTH-1:0] test_f1 = test_features[DATA_WIDTH-1:0];
    
    // ── Training Data (same as main classifier) ──────────────────────────
    reg [DATA_WIDTH-1:0]  training_f0 [0:NUM_TRAINING-1];
    reg [DATA_WIDTH-1:0]  training_f1 [0:NUM_TRAINING-1];
    
    // Initialize training data
    initial begin
        training_f0[ 0] = 8'd 55;  training_f1[ 0] = 8'd 73;
        training_f0[ 1] = 8'd 50;  training_f1[ 1] = 8'd169;
        training_f0[ 2] = 8'd 43;  training_f1[ 2] = 8'd 74;
        training_f0[ 3] = 8'd 69;  training_f1[ 3] = 8'd 78;
        training_f0[ 4] = 8'd 52;  training_f1[ 4] = 8'd 28;
        training_f0[ 5] = 8'd  9;  training_f1[ 5] = 8'd136;
        training_f0[ 6] = 8'd 57;  training_f1[ 6] = 8'd 42;
        training_f0[ 7] = 8'd155;  training_f1[ 7] = 8'd 90;
        training_f0[ 8] = 8'd 91;  training_f1[ 8] = 8'd111;
        training_f0[ 9] = 8'd 66;  training_f1[ 9] = 8'd 26;
        training_f0[10] = 8'd181;  training_f1[10] = 8'd105;
        training_f0[11] = 8'd100;  training_f1[11] = 8'd134;
        training_f0[12] = 8'd159;  training_f1[12] = 8'd 84;
        training_f0[13] = 8'd 88;  training_f1[13] = 8'd 61;
        training_f0[14] = 8'd 69;  training_f1[14] = 8'd 31;
        training_f0[15] = 8'd 78;  training_f1[15] = 8'd 95;
        training_f0[16] = 8'd 97;  training_f1[16] = 8'd 87;
        training_f0[17] = 8'd 96;  training_f1[17] = 8'd 86;
        training_f0[18] = 8'd 70;  training_f1[18] = 8'd 71;
        training_f0[19] = 8'd 44;  training_f1[19] = 8'd 46;
        training_f0[20] = 8'd117;  training_f1[20] = 8'd 90;
        training_f0[21] = 8'd 61;  training_f1[21] = 8'd 32;
        training_f0[22] = 8'd 67;  training_f1[22] = 8'd 57;
        training_f0[23] = 8'd 66;  training_f1[23] = 8'd 61;
        training_f0[24] = 8'd 91;  training_f1[24] = 8'd 36;
        training_f0[25] = 8'd 66;  training_f1[25] = 8'd 51;
        training_f0[26] = 8'd 33;  training_f1[26] = 8'd159;
        training_f0[27] = 8'd 69;  training_f1[27] = 8'd 68;
        training_f0[28] = 8'd 30;  training_f1[28] = 8'd 23;
        training_f0[29] = 8'd 69;  training_f1[29] = 8'd170;
        training_f0[30] = 8'd 79;  training_f1[30] = 8'd 40;
        training_f0[31] = 8'd 59;  training_f1[31] = 8'd 74;
        training_f0[32] = 8'd133;  training_f1[32] = 8'd104;
        training_f0[33] = 8'd122;  training_f1[33] = 8'd 57;
        training_f0[34] = 8'd 51;  training_f1[34] = 8'd 31;
        training_f0[35] = 8'd 72;  training_f1[35] = 8'd132;
        training_f0[36] = 8'd 38;  training_f1[36] = 8'd 44;
        training_f0[37] = 8'd 93;  training_f1[37] = 8'd 36;
        training_f0[38] = 8'd 66;  training_f1[38] = 8'd 57;
        training_f0[39] = 8'd 75;  training_f1[39] = 8'd 89;
        training_f0[40] = 8'd 31;  training_f1[40] = 8'd 53;
        training_f0[41] = 8'd 78;  training_f1[41] = 8'd 74;
        training_f0[42] = 8'd 71;  training_f1[42] = 8'd165;
        training_f0[43] = 8'd220;  training_f1[43] = 8'd131;
        training_f0[44] = 8'd 92;  training_f1[44] = 8'd 51;
        training_f0[45] = 8'd 51;  training_f1[45] = 8'd 28;
        training_f0[46] = 8'd 62;  training_f1[46] = 8'd 71;
        training_f0[47] = 8'd124;  training_f1[47] = 8'd135;
        training_f0[48] = 8'd123;  training_f1[48] = 8'd127;
        training_f0[49] = 8'd 89;  training_f1[49] = 8'd 86;
    end
    
    // ── Distance Calculation Combinational Logic ─────────────────────────
    wire [9:0] abs_diff0 = (test_f0 > training_f0[sample_idx]) ?
                           (test_f0 - training_f0[sample_idx]) :
                           (training_f0[sample_idx] - test_f0);
    
    wire [9:0] abs_diff1 = (test_f1 > training_f1[sample_idx]) ?
                           (test_f1 - training_f1[sample_idx]) :
                           (training_f1[sample_idx] - test_f1);
    
    wire [DIST_WIDTH-1:0] manhattan_dist = abs_diff0 + abs_diff1;
    wire [DIST_WIDTH-1:0] euclidean_dist = (abs_diff0 * abs_diff0) + (abs_diff1 * abs_diff1);
    wire [DIST_WIDTH-1:0] current_distance = (DISTANCE_METRIC == 0) ? manhattan_dist : euclidean_dist;
    
    // ── Sequential Logic ────────────────────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            sample_idx <= 0;
            progress <= 0;
            distances_valid <= 0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    sample_idx <= 0;
                    progress <= 0;
                    distances_valid <= 0;
                end
                
                CALCULATING: begin
                    if (sample_idx < NUM_TRAINING) begin
                        distances[sample_idx] <= current_distance;
                        sample_idx <= sample_idx + 1;
                        progress <= (sample_idx * 8'd255) / NUM_TRAINING;  // 0-255 progress indicator
                    end
                end
                
                DONE: begin
                    distances_valid <= 1;
                    progress <= 8'd255;
                end
            endcase
        end
    end
    
    // ── Combinational Next-State Logic ──────────────────────────────────
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:        if (start)                   next_state = CALCULATING;
            CALCULATING: if (sample_idx >= NUM_TRAINING) next_state = DONE;
            DONE:        if (!start)                  next_state = IDLE;
            default:     next_state = IDLE;
        endcase
    end

endmodule
