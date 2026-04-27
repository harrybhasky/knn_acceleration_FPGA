module knn_classifier #(
    parameter DATA_WIDTH = 8,
    parameter NUM_FEATURES = 2,
    parameter NUM_TRAINING = 50,
    parameter K_VALUE = 3
)(
    input wire clk,
    input wire rst,
    input wire start,
    input wire [DATA_WIDTH*NUM_FEATURES-1:0] test_data,
    input wire [DATA_WIDTH-1:0] k_value,
    output reg predicted_class,
    output reg done
);

    localparam IDLE          = 2'd0;
    localparam CALC_DISTANCE = 2'd1;
    localparam CLASSIFY      = 2'd2;
    localparam DONE_STATE    = 2'd3;

    reg [1:0] state;
    reg [6:0] i_reg;
    reg [7:0] k_reg;

    reg [DATA_WIDTH-1:0] training_data1 [0:NUM_TRAINING-1];
    reg [DATA_WIDTH-1:0] training_data2 [0:NUM_TRAINING-1];
    reg training_label [0:NUM_TRAINING-1];

    reg [DATA_WIDTH-1:0] test_feature1, test_feature2;
    reg [DATA_WIDTH+1:0] curr_distance;

    // Keep only the top-3 nearest neighbors while streaming distances.
    reg [DATA_WIDTH+1:0] nearest_dist0, nearest_dist1, nearest_dist2;
    reg nearest_label0, nearest_label1, nearest_label2;
    reg [2:0] vote_class0, vote_class1;

    function [DATA_WIDTH:0] abs_diff;
        input [DATA_WIDTH-1:0] a;
        input [DATA_WIDTH-1:0] b;
        begin
            if (a >= b) abs_diff = a - b;
            else        abs_diff = b - a;
        end
    endfunction

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

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            i_reg <= 0;
            k_reg <= K_VALUE;
            predicted_class <= 1'b0;
            done <= 1'b0;
            test_feature1 <= 0;
            test_feature2 <= 0;
            curr_distance <= 0;
            nearest_dist0 <= {DATA_WIDTH+2{1'b1}};
            nearest_dist1 <= {DATA_WIDTH+2{1'b1}};
            nearest_dist2 <= {DATA_WIDTH+2{1'b1}};
            nearest_label0 <= 1'b0;
            nearest_label1 <= 1'b0;
            nearest_label2 <= 1'b0;
            vote_class0 <= 0;
            vote_class1 <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        test_feature1 <= test_data[15:8];
                        test_feature2 <= test_data[7:0];
                        k_reg <= (k_value > 0 && k_value <= K_VALUE) ? k_value : K_VALUE;
                        i_reg <= 0;
                        nearest_dist0 <= {DATA_WIDTH+2{1'b1}};
                        nearest_dist1 <= {DATA_WIDTH+2{1'b1}};
                        nearest_dist2 <= {DATA_WIDTH+2{1'b1}};
                        nearest_label0 <= 1'b0;
                        nearest_label1 <= 1'b0;
                        nearest_label2 <= 1'b0;
                        state <= CALC_DISTANCE;
                    end
                end

                CALC_DISTANCE: begin
                    if (i_reg < NUM_TRAINING) begin
                        curr_distance = abs_diff(test_feature1, training_data1[i_reg]) +
                                        abs_diff(test_feature2, training_data2[i_reg]);

                        if (curr_distance < nearest_dist0) begin
                            nearest_dist2  <= nearest_dist1;
                            nearest_label2 <= nearest_label1;
                            nearest_dist1  <= nearest_dist0;
                            nearest_label1 <= nearest_label0;
                            nearest_dist0  <= curr_distance;
                            nearest_label0 <= training_label[i_reg];
                        end else if (curr_distance < nearest_dist1) begin
                            nearest_dist2  <= nearest_dist1;
                            nearest_label2 <= nearest_label1;
                            nearest_dist1  <= curr_distance;
                            nearest_label1 <= training_label[i_reg];
                        end else if (curr_distance < nearest_dist2) begin
                            nearest_dist2  <= curr_distance;
                            nearest_label2 <= training_label[i_reg];
                        end

                        i_reg <= i_reg + 1;
                    end else begin
                        state <= CLASSIFY;
                    end
                end

                CLASSIFY: begin
                    vote_class0 = 0;
                    vote_class1 = 0;

                    if (k_reg >= 1) begin
                        if (nearest_label0 == 1'b0) vote_class0 = vote_class0 + 1;
                        else                        vote_class1 = vote_class1 + 1;
                    end
                    if (k_reg >= 2) begin
                        if (nearest_label1 == 1'b0) vote_class0 = vote_class0 + 1;
                        else                        vote_class1 = vote_class1 + 1;
                    end
                    if (k_reg >= 3) begin
                        if (nearest_label2 == 1'b0) vote_class0 = vote_class0 + 1;
                        else                        vote_class1 = vote_class1 + 1;
                    end

                    predicted_class <= (vote_class1 > vote_class0) ? 1'b1 : 1'b0;
                    done <= 1'b1;
                    state <= DONE_STATE;
                end

                DONE_STATE: begin
                    if (!start) begin
                        done <= 1'b0;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule
