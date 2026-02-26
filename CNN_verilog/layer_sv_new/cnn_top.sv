module cnn_top(
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in,
    input logic valid_in,
    input logic signed[15:0] weight_b1 [0:3][0:8],
    input logic signed[31:0] bias_b1 [0:3],
    input logic signed[15:0] weight_b2 [0:3][0:3][0:8],
    input logic signed[31:0] bias_b2 [0:3],
    input logic signed[15:0] weight_b3 [0:3][0:3][0:8],
    input logic signed[31:0] bias_b3 [0:3],
    input logic signed[15:0] weight_b4 [0:3][0:3][0:8],
    input logic signed[31:0] bias_b4 [0:3],
    input logic signed[15:0] weight_class [0:9][0:3][0:48],
    input logic signed[31:0] bias_class [0:9],
    output logic[3:0] result,
    output logic ready,
    output logic done
);

logic signed[15:0] pixel_out_cnn [0:3];
logic[3:0] valid_out_cnn;

conv_top_b4 cnn(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in),
    .valid_in(valid_in),
    .weight_b1(weight_b1),
    .weight_b2(weight_b2),
    .weight_b3(weight_b3),
    .weight_b4(weight_b4),
    .bias_b1(bias_b1),
    .bias_b2(bias_b2),
    .bias_b3(bias_b3),
    .bias_b4(bias_b4),
    .pixel_out(pixel_out_cnn),
    .ready(ready),
    .valid_out(valid_out_cnn)
);

classifier classify(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_out_cnn),
    .valid_in(valid_out_cnn[0]),
    .weight(weight_class),
    .bias(bias_class),
    .result(result),
    .done(done)
);


endmodule