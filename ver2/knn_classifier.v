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

    localparam IDLE = 3'd0;
    localparam CALC_DISTANCE = 3'd1;
    localparam FIND_MIN_INIT = 3'd2;
    localparam FIND_MIN = 3'd3;
    localparam MARK_MIN = 3'd4;
    localparam CLASSIFY = 3'd5;
    localparam DONE_STATE = 3'd6;

    reg [2:0] state;
    reg [6:0] i_reg;
    reg [6:0] k_counter;
    reg [6:0] min_idx;
    reg [12:0] min_dist;
    
    reg [DATA_WIDTH-1:0] training_data1 [0:NUM_TRAINING-1];
    reg [DATA_WIDTH-1:0] training_data2 [0:NUM_TRAINING-1];
    reg training_label [0:NUM_TRAINING-1];
    reg [12:0] training_distance [0:NUM_TRAINING-1];
    reg used [0:NUM_TRAINING-1];
    
    reg [DATA_WIDTH-1:0] test_feature1, test_feature2;
    reg [6:0] k_reg;
    
    reg [6:0] freq_class0, freq_class1;
    
    integer idx;
    initial begin
        training_data1[0] = 8'd1;   training_data2[0] = 8'd12;  training_label[0] = 0;
        training_data1[1] = 8'd2;   training_data2[1] = 8'd5;   training_label[1] = 0;
        training_data1[2] = 8'd5;   training_data2[2] = 8'd3;   training_label[2] = 1;
        training_data1[3] = 8'd3;   training_data2[3] = 8'd2;   training_label[3] = 1;
        training_data1[4] = 8'd3;   training_data2[4] = 8'd6;   training_label[4] = 0;
        training_data1[5] = 8'd7;   training_data2[5] = 8'd8;   training_label[5] = 1;
        training_data1[6] = 8'd8;   training_data2[6] = 8'd2;   training_label[6] = 1;
        training_data1[7] = 8'd1;   training_data2[7] = 8'd9;   training_label[7] = 0;
        training_data1[8] = 8'd4;   training_data2[8] = 8'd7;   training_label[8] = 0;
        training_data1[9] = 8'd6;   training_data2[9] = 8'd4;   training_label[9] = 1;
        
        for (idx = 10; idx < NUM_TRAINING; idx = idx + 1) begin
            training_data1[idx] = 8'd5;
            training_data2[idx] = 8'd5;
            training_label[idx] = idx[0];
        end
    end

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            i_reg <= 0;
            k_counter <= 0;
            k_reg <= K_VALUE;
            predicted_class <= 0;
            done <= 0;
            freq_class0 <= 0;
            freq_class1 <= 0;
            min_idx <= 0;
            min_dist <= 13'h1FFF;
            test_feature1 <= 0;
            test_feature2 <= 0;
            for (idx = 0; idx < NUM_TRAINING; idx = idx + 1)
                used[idx] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        test_feature1 <= test_data[15:8];
                        test_feature2 <= test_data[7:0];
                        k_reg <= (k_value > 0 && k_value <= NUM_TRAINING) ? k_value : K_VALUE;
                        i_reg <= 0;
                        k_counter <= 0;
                        freq_class0 <= 0;
                        freq_class1 <= 0;
                        for (idx = 0; idx < NUM_TRAINING; idx = idx + 1)
                            used[idx] <= 0;
                        state <= CALC_DISTANCE;
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
                        i_reg <= 0;
                        state <= FIND_MIN_INIT;
                    end
                end
                
                FIND_MIN_INIT: begin
                    min_dist <= 13'h1FFF;
                    min_idx <= 0;
                    i_reg <= 0;
                    state <= FIND_MIN;
                end
                
                FIND_MIN: begin
                    if (i_reg < NUM_TRAINING) begin
                        if (!used[i_reg] && training_distance[i_reg] < min_dist) begin
                            min_dist <= training_distance[i_reg];
                            min_idx <= i_reg;
                        end
                        i_reg <= i_reg + 1;
                    end else begin
                        state <= MARK_MIN;
                    end
                end
                
                MARK_MIN: begin
                    used[min_idx] <= 1;
                    if (training_label[min_idx] == 0)
                        freq_class0 <= freq_class0 + 1;
                    else
                        freq_class1 <= freq_class1 + 1;
                    
                    k_counter <= k_counter + 1;
                    
                    if (k_counter + 1 >= k_reg)
                        state <= CLASSIFY;
                    else
                        state <= FIND_MIN_INIT;
                end
                
                CLASSIFY: begin
                    predicted_class <= (freq_class1 > freq_class0) ? 1 : 0;
                    done <= 1;
                    state <= DONE_STATE;
                end
                
                DONE_STATE: begin
                    if (!start) begin
                        done <= 0;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end

endmodule