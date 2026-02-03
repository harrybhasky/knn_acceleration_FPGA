module tb_knn;

    reg clk, rst, start;
    reg [7:0] test_data;
    wire [7:0] best_label;
    wire done;

    knn_top DUT (
        .clk(clk),
        .rst(rst),
        .start(start),
        .test_data(test_data),
        .best_label(best_label),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst = 1;
        start = 0;
        test_data = 0;

        #10 rst = 0;

        #10 test_data = 8'd28;
        start = 1;

        #100 start = 0;

        #50 test_data = 8'd85;
        rst = 1;

        #10 rst = 0;
        start = 1;

        #100 $finish;
    end
endmodule
