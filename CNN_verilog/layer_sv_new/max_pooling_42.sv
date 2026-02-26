module max_pooling_4_2 (
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in [0:3],
    input logic[3:0] valid_in [0:3],
    output logic signed[15:0] pixel_out [0:3],
    output logic[3:0] valid_out
);

logic[3:0] valid_channels;
assign valid_channels[0] = (valid_in[0]==4'b1111);
assign valid_channels[1] = (valid_in[1]==4'b1111);
assign valid_channels[2] = (valid_in[2]==4'b1111);
assign valid_channels[3] = (valid_in[3]==4'b1111);

max_pooling #(.IMG_SIZE(14)) channel_1(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in[0]),
    .valid_in(valid_channels[0]),
    .pixel_out(pixel_out[0]),
    .valid_out(valid_out[0])
);

max_pooling #(.IMG_SIZE(14)) channel_2(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in[1]),
    .valid_in(valid_channels[1]),
    .pixel_out(pixel_out[1]),
    .valid_out(valid_out[1])
);

max_pooling #(.IMG_SIZE(14)) channel_3(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in[2]),
    .valid_in(valid_channels[2]),
    .pixel_out(pixel_out[2]),
    .valid_out(valid_out[2])
);

max_pooling #(.IMG_SIZE(14)) channel_4(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in[3]),
    .valid_in(valid_channels[3]),
    .pixel_out(pixel_out[3]),
    .valid_out(valid_out[3])
);


endmodule