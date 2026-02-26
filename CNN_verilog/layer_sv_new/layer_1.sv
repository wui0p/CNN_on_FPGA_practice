module layer_1 (
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in,
    input logic valid_in,
    input logic signed[15:0] weight_b1 [0:3][0:8],
    input logic signed[31:0] bias_b1 [0:3],
    output logic ready,
    output logic signed[15:0] pixel_out [0:3],
    output logic[3:0] valid_out
);

logic[3:0] ready_channels;
logic[9:0] pixel_now [0:3];

assign ready = (ready_channels==4'b1111);

conv_layer_28 channel_1(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in),
    .valid_in(valid_in),
    .weight(weight_b1[0]),
    .bias(bias_b1[0]),
    .ready(ready_channels[0]),
    .pixel_out(pixel_out[0]),
    .valid_out(valid_out[0])
);

conv_layer_28 channel_2(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in),
    .valid_in(valid_in),
    .weight(weight_b1[1]),
    .bias(bias_b1[1]),
    .ready(ready_channels[1]),
    .pixel_out(pixel_out[1]),
    .valid_out(valid_out[1])
);

conv_layer_28 channel_3(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in),
    .valid_in(valid_in),
    .weight(weight_b1[2]),
    .bias(bias_b1[2]),
    .ready(ready_channels[2]),
    .pixel_out(pixel_out[2]),
    .valid_out(valid_out[2])
);

conv_layer_28 channel_4(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in),
    .valid_in(valid_in),
    .weight(weight_b1[3]),
    .bias(bias_b1[3]),
    .ready(ready_channels[3]),
    .pixel_out(pixel_out[3]),
    .valid_out(valid_out[3])
);



always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        pixel_now[0] <= 0;
        pixel_now[1] <= 0;
        pixel_now[2] <= 0;
        pixel_now[3] <= 0;
    end else begin
        if(valid_out[0]) pixel_now[0] <= pixel_now[0] + 1;
        if(valid_out[1]) pixel_now[1] <= pixel_now[1] + 1;
        if(valid_out[2]) pixel_now[2] <= pixel_now[2] + 1;
        if(valid_out[3]) pixel_now[3] <= pixel_now[3] + 1;
    end
end


endmodule