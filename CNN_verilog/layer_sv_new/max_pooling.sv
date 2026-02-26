module max_pooling #(
    parameter IMG_SIZE = 28
)(
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in,
    input logic valid_in,
    output logic signed[15:0] pixel_out,
    output logic valid_out
);

logic signed[15:0] save_row [0:IMG_SIZE-1];
logic signed[15:0] prev_pixel;
logic[5:0] x_cnt, y_cnt;
logic signed[15:0] max1, max2, max;

//find max
always_comb begin
    if(x_cnt[0] && y_cnt[0]) begin
        max1 = 0;
        max2 = 0;
        max = 0;
        if(save_row[x_cnt] > save_row[x_cnt - 1]) max1 = save_row[x_cnt];
        else max1 = save_row[x_cnt - 1];
        if(pixel_in > prev_pixel) max2 = pixel_in;
        else max2 = prev_pixel;
        if(max1 > max2) max = max1;
        else max = max2;
    end
end


//count
always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        x_cnt <= 0;
        y_cnt <= 0;
    end else if(valid_in) begin
        if(x_cnt == IMG_SIZE - 1) begin
            x_cnt <= 0;
            if(y_cnt == IMG_SIZE - 1) y_cnt <= 0;
            else y_cnt <= y_cnt + 1;
        end else begin
            x_cnt <= x_cnt + 1;
        end
    end
end



//save & pixel
always_ff @(posedge clk) begin
    valid_out <= 0;
    if(valid_in) begin
        if(y_cnt[0] == 0) save_row[x_cnt] <= pixel_in;    //odd rows
        else begin
            if(x_cnt[0] == 0) prev_pixel <= pixel_in;   //pixel still at the bottom left
            else begin
                pixel_out <= max;
                valid_out <= 1;
            end
        end
    end
end


endmodule