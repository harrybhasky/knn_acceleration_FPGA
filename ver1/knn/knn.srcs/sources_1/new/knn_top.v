module knn_top (
    input        clk,
    input        rst,
    input        start,
    input  [7:0] test_data,
    output reg [7:0] best_label,
    output reg       done
);

    reg [2:0] index;
    reg [8:0] min_dist;
    reg [7:0] train_mem [0:4];
    reg [7:0] label_mem [0:4];

    wire [8:0] curr_dist;

    distance_calc D0 (
        .a(test_data),
        .b(train_mem[index]),
        .dist(curr_dist)
    );

    initial begin
        train_mem[0] = 8'd10; label_mem[0] = 8'd1;
        train_mem[1] = 8'd25; label_mem[1] = 8'd2;
        train_mem[2] = 8'd40; label_mem[2] = 8'd3;
        train_mem[3] = 8'd60; label_mem[3] = 8'd4;
        train_mem[4] = 8'd90; label_mem[4] = 8'd5;
    end

    always @(posedge clk) begin
        if (rst) begin
            index <= 0;
            min_dist <= 9'h1FF;
            best_label <= 0;
            done <= 0;
        end else if (start && !done) begin
            if (curr_dist < min_dist) begin
                min_dist <= curr_dist;
                best_label <= label_mem[index];
            end

            if (index == 4) begin
                done <= 1;
            end else begin
                index <= index + 1;
            end
        end
    end
endmodule
