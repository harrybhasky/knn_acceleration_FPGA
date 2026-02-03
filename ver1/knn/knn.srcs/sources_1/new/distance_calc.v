module distance_calc (
    input  [7:0] a,
    input  [7:0] b,
    output [8:0] dist
);
    assign dist = (a > b) ? (a - b) : (b - a);
endmodule
