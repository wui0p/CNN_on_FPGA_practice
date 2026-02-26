module padding_control #(
    parameter IMG_SIZE = 28
)(
    input logic clk,
    input logic reset,
    input logic signed[15:0] data_in,
    input logic valid_in,
    output logic ready,
    output logic signed[15:0] data_out,
    output logic valid_out
);

localparam PAD_SIZE = IMG_SIZE + 2;
logic[5:0] x_cnt, y_cnt;
logic is_padding;
logic is_region;

always_comb begin
    is_padding = (x_cnt==0 || x_cnt==PAD_SIZE-1 || y_cnt==0 || y_cnt==PAD_SIZE-1);
    is_region = ~is_padding;
end

always_comb begin
    if(is_padding) begin
        data_out = 0;
        ready = 0;
        valid_out = 1;
    end else begin
        data_out = data_in;
        ready = 1;
        valid_out = valid_in;
    end
end

always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        x_cnt <= 0;
        y_cnt <= 0;
    end else if(is_padding || (is_region && valid_in)) begin
        if(x_cnt==PAD_SIZE-1) begin
            x_cnt <= 0;
            if(y_cnt==PAD_SIZE-1) y_cnt <= 0;
            else y_cnt <= y_cnt + 1;
        end else x_cnt <= x_cnt + 1;
    end
end

endmodule