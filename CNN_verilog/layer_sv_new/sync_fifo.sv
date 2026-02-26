module sync_fifo(
    input logic clk,
    input logic reset,
    input logic wr_en,
    input logic rd_en,
    input logic signed[15:0] data_in [0:3],
    output logic full,
    output logic empty,
    output logic signed[15:0] data_out [0:3]
);

logic signed[15:0] mem [0:127][0:3];
logic[6:0] wr_cnt, rd_cnt;
logic[7:0] count;
logic read_now, write_now;

assign empty = (count==0);
assign full = (count==128);
assign data_out = mem[rd_cnt];

assign write_now = wr_en && !full;
assign read_now = rd_en && !empty;

always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        wr_cnt <= 0;
        rd_cnt <= 0;
        count <= 0;
    end else begin
        if(write_now) begin
            mem[wr_cnt] <= data_in;
            wr_cnt <= wr_cnt + 1;
        end
        if(read_now) rd_cnt <= rd_cnt + 1;

        if(write_now && !read_now) count <= count + 1;
        else if(!write_now && read_now) count <= count - 1;
    end
end

endmodule