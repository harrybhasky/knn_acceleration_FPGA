// ============================================================================
// K-Nearest Neighbors (k-NN) Classifier — k=3
// Dataset : Breast Cancer Wisconsin (2 features, 8-bit scaled, 50 samples)
// Feature 0: mean radius  (scaled 0–255)
// Feature 1: mean texture (scaled 0–255)
// Distance : Manhattan
// ============================================================================

module knn_classifier #(
    parameter DATA_WIDTH   = 8,
    parameter NUM_FEATURES = 2,
    parameter NUM_TRAINING = 50,
    parameter K_VALUE      = 3
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire [DATA_WIDTH*NUM_FEATURES-1:0] test_data,
    input  wire [DATA_WIDTH-1:0] k_value,
    output reg  predicted_class,
    output reg  done
);

    // ── State Encoding ──────────────────────────────────────────────────────
    localparam IDLE         = 4'd0;
    localparam LOAD_DATA    = 4'd1;
    localparam CALC_DISTANCE= 4'd2;
    localparam SORT_START   = 4'd3;
    localparam SORT_OUTER   = 4'd4;
    localparam SORT_INNER   = 4'd5;
    localparam SORT_SWAP    = 4'd6;
    localparam CLASSIFY     = 4'd7;
    localparam DONE         = 4'd8;

    reg [3:0] state, next_state;
    reg [6:0] i_reg, j_reg, k_reg;

    // ── Memory ───────────────────────────────────────────────────────────────
    reg [DATA_WIDTH-1:0]   training_data1  [0:NUM_TRAINING-1];
    reg [DATA_WIDTH-1:0]   training_data2  [0:NUM_TRAINING-1];
    reg                    training_label  [0:NUM_TRAINING-1];
    reg [DATA_WIDTH+4:0]   training_distance[0:NUM_TRAINING-1];

    reg [DATA_WIDTH+4:0]   temp_distance;
    reg                    temp_label;
    reg [DATA_WIDTH-1:0]   temp_data1, temp_data2;

    reg [DATA_WIDTH-1:0]   test_feature1, test_feature2;
    reg [6:0]              freq_class0, freq_class1;

    // ── Training Data — Breast Cancer Wisconsin (50 samples, 8-bit scaled) ──
    integer idx;
    initial begin
        training_data1[ 0] = 8'd 55;  training_data2[ 0] = 8'd 73;  training_label[ 0] = 1;
        training_data1[ 1] = 8'd 50;  training_data2[ 1] = 8'd169;  training_label[ 1] = 1;
        training_data1[ 2] = 8'd 43;  training_data2[ 2] = 8'd 74;  training_label[ 2] = 1;
        training_data1[ 3] = 8'd 69;  training_data2[ 3] = 8'd 78;  training_label[ 3] = 1;
        training_data1[ 4] = 8'd 52;  training_data2[ 4] = 8'd 28;  training_label[ 4] = 1;
        training_data1[ 5] = 8'd  9;  training_data2[ 5] = 8'd136;  training_label[ 5] = 1;
        training_data1[ 6] = 8'd 57;  training_data2[ 6] = 8'd 42;  training_label[ 6] = 1;
        training_data1[ 7] = 8'd155;  training_data2[ 7] = 8'd 90;  training_label[ 7] = 0;
        training_data1[ 8] = 8'd 91;  training_data2[ 8] = 8'd111;  training_label[ 8] = 1;
        training_data1[ 9] = 8'd 66;  training_data2[ 9] = 8'd 26;  training_label[ 9] = 1;
        training_data1[10] = 8'd181;  training_data2[10] = 8'd105;  training_label[10] = 0;
        training_data1[11] = 8'd100;  training_data2[11] = 8'd134;  training_label[11] = 0;
        training_data1[12] = 8'd159;  training_data2[12] = 8'd 84;  training_label[12] = 0;
        training_data1[13] = 8'd 88;  training_data2[13] = 8'd 61;  training_label[13] = 1;
        training_data1[14] = 8'd 69;  training_data2[14] = 8'd 31;  training_label[14] = 1;
        training_data1[15] = 8'd 78;  training_data2[15] = 8'd 95;  training_label[15] = 0;
        training_data1[16] = 8'd 97;  training_data2[16] = 8'd 87;  training_label[16] = 0;
        training_data1[17] = 8'd 96;  training_data2[17] = 8'd 86;  training_label[17] = 1;
        training_data1[18] = 8'd 70;  training_data2[18] = 8'd 71;  training_label[18] = 1;
        training_data1[19] = 8'd 44;  training_data2[19] = 8'd 46;  training_label[19] = 1;
        training_data1[20] = 8'd117;  training_data2[20] = 8'd 90;  training_label[20] = 0;
        training_data1[21] = 8'd 61;  training_data2[21] = 8'd 32;  training_label[21] = 1;
        training_data1[22] = 8'd 67;  training_data2[22] = 8'd 57;  training_label[22] = 1;
        training_data1[23] = 8'd 66;  training_data2[23] = 8'd 61;  training_label[23] = 1;
        training_data1[24] = 8'd 91;  training_data2[24] = 8'd 36;  training_label[24] = 1;
        training_data1[25] = 8'd 66;  training_data2[25] = 8'd 51;  training_label[25] = 0;
        training_data1[26] = 8'd 33;  training_data2[26] = 8'd159;  training_label[26] = 1;
        training_data1[27] = 8'd 69;  training_data2[27] = 8'd 68;  training_label[27] = 1;
        training_data1[28] = 8'd 30;  training_data2[28] = 8'd 23;  training_label[28] = 1;
        training_data1[29] = 8'd 69;  training_data2[29] = 8'd170;  training_label[29] = 1;
        training_data1[30] = 8'd 79;  training_data2[30] = 8'd 40;  training_label[30] = 1;
        training_data1[31] = 8'd 59;  training_data2[31] = 8'd 74;  training_label[31] = 1;
        training_data1[32] = 8'd133;  training_data2[32] = 8'd104;  training_label[32] = 0;
        training_data1[33] = 8'd122;  training_data2[33] = 8'd 57;  training_label[33] = 0;
        training_data1[34] = 8'd 51;  training_data2[34] = 8'd 31;  training_label[34] = 1;
        training_data1[35] = 8'd 72;  training_data2[35] = 8'd132;  training_label[35] = 1;
        training_data1[36] = 8'd 38;  training_data2[36] = 8'd 44;  training_label[36] = 1;
        training_data1[37] = 8'd 93;  training_data2[37] = 8'd 36;  training_label[37] = 1;
        training_data1[38] = 8'd 66;  training_data2[38] = 8'd 57;  training_label[38] = 1;
        training_data1[39] = 8'd 75;  training_data2[39] = 8'd 89;  training_label[39] = 1;
        training_data1[40] = 8'd 31;  training_data2[40] = 8'd 53;  training_label[40] = 1;
        training_data1[41] = 8'd 78;  training_data2[41] = 8'd 74;  training_label[41] = 1;
        training_data1[42] = 8'd 71;  training_data2[42] = 8'd165;  training_label[42] = 1;
        training_data1[43] = 8'd220;  training_data2[43] = 8'd131;  training_label[43] = 0;
        training_data1[44] = 8'd 92;  training_data2[44] = 8'd 51;  training_label[44] = 1;
        training_data1[45] = 8'd 51;  training_data2[45] = 8'd 28;  training_label[45] = 1;
        training_data1[46] = 8'd 62;  training_data2[46] = 8'd 71;  training_label[46] = 1;
        training_data1[47] = 8'd124;  training_data2[47] = 8'd135;  training_label[47] = 0;
        training_data1[48] = 8'd123;  training_data2[48] = 8'd127;  training_label[48] = 0;
        training_data1[49] = 8'd 89;  training_data2[49] = 8'd 86;  training_label[49] = 0;
    end

    // ── Sequential Logic ─────────────────────────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;  i_reg <= 0;  j_reg <= 0;
            k_reg <= K_VALUE;  predicted_class <= 0;  done <= 0;
            freq_class0 <= 0;  freq_class1 <= 0;
            test_feature1 <= 0;  test_feature2 <= 0;
        end else begin
            state <= next_state;
            case (state)
                IDLE: begin
                    if (start) begin
                        test_feature1 <= test_data[DATA_WIDTH*2-1:DATA_WIDTH];
                        test_feature2 <= test_data[DATA_WIDTH-1:0];
                        k_reg <= (k_value > 0 && k_value <= NUM_TRAINING) ? k_value : K_VALUE;
                        i_reg <= 0;  done <= 0;
                    end
                end
                CALC_DISTANCE: begin
                    if (i_reg < NUM_TRAINING) begin
                        training_distance[i_reg] <=
                            ((test_feature1 > training_data1[i_reg]) ?
                             (test_feature1 - training_data1[i_reg]) :
                             (training_data1[i_reg] - test_feature1)) +
                            ((test_feature2 > training_data2[i_reg]) ?
                             (test_feature2 - training_data2[i_reg]) :
                             (training_data2[i_reg] - test_feature2));
                        i_reg <= i_reg + 1;
                    end else begin
                        i_reg <= 0;  j_reg <= 0;
                    end
                end
                SORT_OUTER: begin
                    if (i_reg < NUM_TRAINING - 1) j_reg <= 0;
                    else i_reg <= 0;
                end
                SORT_INNER: begin
                    if (j_reg < NUM_TRAINING - i_reg - 1) begin
                        if (training_distance[j_reg] > training_distance[j_reg + 1]) begin
                            temp_distance <= training_distance[j_reg];
                            temp_label    <= training_label[j_reg];
                            temp_data1    <= training_data1[j_reg];
                            temp_data2    <= training_data2[j_reg];
                        end
                    end else begin
                        i_reg <= i_reg + 1;
                    end
                end
                SORT_SWAP: begin
                    training_distance[j_reg]   <= training_distance[j_reg + 1];
                    training_label[j_reg]       <= training_label[j_reg + 1];
                    training_data1[j_reg]       <= training_data1[j_reg + 1];
                    training_data2[j_reg]       <= training_data2[j_reg + 1];
                    training_distance[j_reg+1]  <= temp_distance;
                    training_label[j_reg+1]     <= temp_label;
                    training_data1[j_reg+1]     <= temp_data1;
                    training_data2[j_reg+1]     <= temp_data2;
                    j_reg <= j_reg + 1;
                end
                CLASSIFY: begin
                    if (i_reg < k_reg) begin
                        if (training_label[i_reg] == 0) freq_class0 <= freq_class0 + 1;
                        else                             freq_class1 <= freq_class1 + 1;
                        i_reg <= i_reg + 1;
                    end else begin
                        predicted_class <= (freq_class1 > freq_class0) ? 1 : 0;
                        done <= 1;
                        freq_class0 <= 0;  freq_class1 <= 0;
                    end
                end
                DONE: begin
                    if (!start) done <= 0;
                end
            endcase
        end
    end

    // ── Combinational Next-State Logic ────────────────────────────────────────
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:          if (start)                          next_state = LOAD_DATA;
            LOAD_DATA:                                         next_state = CALC_DISTANCE;
            CALC_DISTANCE: if (i_reg >= NUM_TRAINING)          next_state = SORT_START;
            SORT_START:                                        next_state = SORT_OUTER;
            SORT_OUTER:    if (i_reg < NUM_TRAINING - 1)       next_state = SORT_INNER;
                           else                                next_state = CLASSIFY;
            SORT_INNER:    if (j_reg < NUM_TRAINING - i_reg - 1) begin
                               if (training_distance[j_reg] > training_distance[j_reg+1])
                                                               next_state = SORT_SWAP;
                               // else stay in SORT_INNER (j increments implicitly via SORT_SWAP exit)
                           end else                            next_state = SORT_OUTER;
            SORT_SWAP:                                         next_state = SORT_INNER;
            CLASSIFY:      if (i_reg >= k_reg)                 next_state = DONE;
            DONE:          if (!start)                         next_state = IDLE;
            default:                                           next_state = IDLE;
        endcase
    end

endmodule
