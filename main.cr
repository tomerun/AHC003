START_TIME = Time.utc.to_unix_ms
TL         = 1800
N          =   30
INF        = 1 << 28
COUNTER    = Counter.new
DIR_CHARS  = "UDLR".chars
DIR_U      = 0
DIR_D      = 1
DIR_L      = 2
DIR_R      = 3
DR         = [-1, 1, 0, 0]
DC         = [0, 0, -1, 1]
XH         = Array.new(N, N)
XV         = Array.new(N, N)

class XorShift
  TO_DOUBLE = 0.5 / (1u64 << 63)

  def initialize(@x = 123456789u64)
  end

  def next_int
    @x ^= @x << 13
    @x ^= @x >> 17
    @x ^= @x << 5
    return @x
  end

  def next_int(m)
    return next_int % m
  end

  def next_double
    return TO_DOUBLE * next_int
  end
end

class Counter
  def initialize
    @hist = [] of Int32
  end

  def add(i)
    while @hist.size <= i
      @hist << 0
    end
    @hist[i] += 1
  end

  def to_s(io)
    io << "counter:\n"
    ((@hist.size + 9) // 10).times do |i|
      io << @hist[((i * 10)...(i * 10 + 10))]
      io << "\n"
    end
  end
end

macro debug(msg)
  {% if flag?(:trace) %}
    STDERR.puts({{msg}})
  {% end %}
end

macro debugf(format_string, *args)
  {% if flag?(:trace) %}
    STDERR.printf({{format_string}}, {{*args}})
  {% end %}
end

class AtCoderJudge
  def query
    ps = read_line.split.map(&.to_i)
    return Tuple(Int32, Int32, Int32, Int32).from(ps)
  end

  def response(path)
    puts path.map { |d| DIR_CHARS[d] }.join
    STDOUT.flush
    return read_line.to_i
  end
end

class LocalJudge
  getter :score
  @horz : Array(Array(Int32))
  @vert : Array(Array(Int32))
  @input_file : File?
  @output_file : File?

  def initialize(@seed : Int32)
    @rnd = Random.new(@seed)
    @score = 0.0
    m = @rnd.rand(2) + 1
    d = @rnd.rand(2000 - 100 + 1) + 100
    if 100 <= @seed && @seed < 200
      m = 1
      d = 100 + (2000 - 100) * (@seed - 100) // 99
    elsif 200 <= @seed && @seed < 300
      m = 2
      d = 100 + (2000 - 100) * (@seed - 200) // 99
    end
    debugf("m:%1d d:%4d\n", m, d)
    @horz = generate_edge(d, m, XH)
    @vert = generate_edge(d, m, XV).transpose
    @sr = 0
    @sc = 0
    @tr = 0
    @tc = 0
    @qi = 0
    {% if flag?(:trace) %}
      @input_file = File.new("out/#{@seed}.in.txt", "w")
      @output_file = File.new("out/#{@seed}.out.txt", "w")
    {% end %}
    if f = @input_file
      N.times do |i|
        f << @horz[i].join(" ") << "\n"
      end
      (N - 1).times do |i|
        f << @vert[i].join(" ") << "\n"
      end
    end

    debug("ground truth")
    @horz.each do |row|
      debug(row.map { |v| sprintf("%4d", v) }.join(" "))
    end
    debug("")
    @vert.transpose.each do |row|
      debug(row.map { |v| sprintf("%4d", v) }.join(" "))
    end
    debug("")
  end

  private def generate_edge(d, m, xs)
    ret = Array.new(N) { [0] * (N - 1) }
    h0s = [] of Int32
    h1s = [] of Int32
    N.times do |i|
      h = Array.new(2) { @rnd.rand(9000 - 1000 - 2 * d + 1) + 1000 + d }
      if m == 1
        h[1] = h[0]
      end
      x = @rnd.rand(N - 2) + 1
      x.times do |j|
        delta = @rnd.rand(2 * d + 1) - d
        ret[i][j] = h[0] + delta
      end
      x.upto(N - 2) do |j|
        delta = @rnd.rand(2 * d + 1) - d
        ret[i][j] = h[1] + delta
      end
      h0s << h[0]
      h1s << h[1]
      xs[i] = x
    end
    debug("e_H0:#{h0s.join(" ")}")
    debug("e_H1:#{h1s.join(" ")}")
    debug("x   :#{xs.map { |v| sprintf("%4d", v) }.join(" ")}")
    return ret
  end

  def query
    while true
      @sr = @rnd.rand(N)
      @sc = @rnd.rand(N)
      @tr = @rnd.rand(N)
      @tc = @rnd.rand(N)
      break if (@sr - @tr).abs + (@sc - @tc).abs >= 10
    end
    return {@sr, @sc, @tr, @tc}
  end

  def response(path)
    dist = 0
    r = @sr
    c = @sc
    path.each do |d|
      case d
      when DIR_U
        dist += @vert[r - 1][c]
        r -= 1
      when DIR_D
        dist += @vert[r][c]
        r += 1
      when DIR_L
        dist += @horz[r][c - 1]
        c -= 1
      when DIR_R
        dist += @horz[r][c]
        c += 1
      end
    end
    if r != @tr || c != @tc
      puts "invalid output #{@qi}"
    end
    sd = calc_shortest_dist
    @score *= 0.998
    @score += sd / dist
    e = @rnd.rand * 0.2 + 0.9
    debugf("qi:%d best:%d actual:%d\n", @qi, sd, dist)
    @qi += 1
    if f = @input_file
      f << @sr << " " << @sc << " " << @tr << " " << @tc << " " << sd << " " << e << "\n"
      if @qi == 1000
        f.close
      end
    end
    if f = @output_file
      f << path.map { |d| DIR_CHARS[d] }.join << "\n"
      if @qi == 1000
        f.close
      end
    end
    return (dist * e).round.to_i
  end

  private def calc_shortest_dist
    q = PriorityQueue(Tuple(Int32, Int32, Int32)).new(N * N)
    q.add({0, @sr, @sc})
    visited = Array.new(N) { [1 << 28] * N }
    visited[@sr][@sc] = 0
    while true
      cur = q.pop
      cd, cr, cc = -cur[0], cur[1], cur[2]
      return cd if cr == @tr && cc == @tc
      if cr != 0
        nd = cd + @vert[cr - 1][cc]
        if nd < visited[cr - 1][cc]
          visited[cr - 1][cc] = nd
          q.add({-nd, cr - 1, cc})
        end
      end
      if cr != N - 1
        nd = cd + @vert[cr][cc]
        if nd < visited[cr + 1][cc]
          visited[cr + 1][cc] = nd
          q.add({-nd, cr + 1, cc})
        end
      end
      if cc != 0
        nd = cd + @horz[cr][cc - 1]
        if nd < visited[cr][cc - 1]
          visited[cr][cc - 1] = nd
          q.add({-nd, cr, cc - 1})
        end
      end
      if cc != N - 1
        nd = cd + @horz[cr][cc]
        if nd < visited[cr][cc + 1]
          visited[cr][cc + 1] = nd
          q.add({-nd, cr, cc + 1})
        end
      end
    end
  end
end

class PriorityQueue(T)
  def initialize(capacity : Int32)
    @elem = Array(T).new(capacity)
  end

  def initialize(list : Enumerable(T))
    @elem = list.to_a
    1.upto(size - 1) { |i| fixup(i) }
  end

  def size
    @elem.size
  end

  def add(v)
    @elem << v
    fixup(size - 1)
  end

  def top
    @elem[0]
  end

  def pop
    ret = @elem[0]
    last = @elem.pop
    if size > 0
      @elem[0] = last
      fixdown(0)
    end
    ret
  end

  def clear
    @elem.clear
  end

  def decrease_top(new_value : T)
    @elem[0] = new_value
    fixdown(0)
  end

  def to_s(io : IO)
    io << @elem
  end

  private def fixup(index : Int32)
    while index > 0
      parent = (index - 1) // 2
      break if @elem[parent] >= @elem[index]
      @elem[parent], @elem[index] = @elem[index], @elem[parent]
      index = parent
    end
  end

  private def fixdown(index : Int32)
    while true
      left = index * 2 + 1
      break if left >= size
      right = index * 2 + 2
      child = right >= size || @elem[left] > @elem[right] ? left : right
      if @elem[child] > @elem[index]
        @elem[child], @elem[index] = @elem[index], @elem[child]
        index = child
      else
        break
      end
    end
  end
end

#################
# end of lib
#################

main

def main
  judge = AtCoderJudge.new
  {% if flag?(:local) %}
    seed = ARGV.empty? ? 1 : ARGV[0].to_i
    judge = LocalJudge.new(seed)
  {% end %}
  solver = Solver.new(judge, START_TIME + TL)
  solver.solve
  {% if flag?(:local) %}
    printf("%.3f\n", judge.score * 2.312311)
    printf("%dms\n", Time.utc.to_unix_ms - START_TIME)
  {% end %}
end

class History
  getter :sr, :sc, :tr, :tc, :path, :b, :hs, :vs
  property :sum_ratio, :dif_hist, :skip_until

  def initialize(@sr : Int32, @sc : Int32, @tr : Int32, @tc : Int32, @path : Array(Int32), @b : Int32)
    @hs = Array(Tuple(Int32, Int32)).new
    @vs = Array(Tuple(Int32, Int32)).new
    @dif_hist = Array(Tuple(Int32, Float64)).new
    @skip_until = 0
    @sum_ratio = 0.0
    cr = @sr
    cc = @sc
    @path.each do |d|
      case d
      when DIR_U
        cr -= 1
        @vs << {cc, cr}
      when DIR_D
        @vs << {cc, cr}
        cr += 1
      when DIR_L
        cc -= 1
        @hs << {cr, cc}
      when DIR_R
        @hs << {cr, cc}
        cc += 1
      end
    end
  end
end

class Solver(Judge)
  @ita : Float64
  @ita2 : Float64

  def initialize(@judge : Judge, @timelimit : Int64)
    @rnd = XorShift.new(2u64)
    @e_horz = Array(Array(Int32)).new(N) do |i|
      [5000] * (N - 1)
    end
    @e_vert = Array(Array(Int32)).new(N) do |i|
      [5000] * (N - 1)
    end
    @c_horz = Array(Array(Int32)).new(N) { [0] * (N - 1) }
    @c_vert = Array(Array(Int32)).new(N) { [0] * (N - 1) }
    @xh = Array(Int32).new(N, N)
    @xv = Array(Int32).new(N, N)
    @sp_visited = Array(Array(Int32)).new(N) { [INF] * N }
    @sp_dir = Array(Array(Int32)).new(N) { [0] * N }
    @sp_q = PriorityQueue(Tuple(Int32, Int32, Int32)).new(N * N)
    @history = Array(History).new
    @qi = 0
    @ita = (ENV["ita"]? || "0.7").to_f
    @ita2 = (ENV["ita2"]? || "0.1").to_f
    @is_m0 = false
  end

  def solve
    pred_freq = (ENV["pred_freq"]? || "5").to_i
    pred_cut = (ENV["pred_cut"]? || "1000").to_i
    1000.times do
      sr, sc, tr, tc = @judge.query
      rev = sr > tr
      if rev
        sr, tr = tr, sr
        sc, tc = tc, sc
      end
      path = select_path(sr, sc, tr, tc)
      rough_dist = @judge.response(rev ? path.reverse.map { |v| v ^ 1 } : path)
      @history << History.new(sr, sc, tr, tc, path, rough_dist)
      @qi += 1
      if @qi < pred_cut && @qi % pred_freq == 0
        predict()
        if @qi % 100 == 0
          debug("predict qi=#{@qi}")
          @e_horz.each do |row|
            debug(row.join(" "))
          end
          debug("")
          @e_vert.each do |row|
            debug(row.join(" "))
          end
          debug("")
          @c_horz.each do |row|
            debug(row.map { |v| sprintf("%3d", v) }.join(" "))
          end
          debug("")
          @c_vert.each do |row|
            debug(row.map { |v| sprintf("%3d", v) }.join(" "))
          end
          debug("")
        end
      else
        postprocess()
      end
    end
  end

  def select_path(sr, sc, tr, tc)
    bonus_unvisited = (ENV["bonus"]? || "1000").to_i + (ENV["bonus_t"]? || "40000").to_i // (@qi + 10)
    th_ave_cost = (ENV["th_ave_b"]? || "2000").to_i + @qi * (ENV["th_ave_m"]? || "4").to_i
    2.times do |li|
      @sp_q.clear
      @sp_q.add({0, sr, sc})
      N.times do |i|
        @sp_visited[i].fill(INF)
      end
      @sp_visited[sr][sc] = 0
      total_dist = 0
      while true
        cur = @sp_q.pop
        cd, cr, cc = -cur[0], cur[1], cur[2]
        if cr == tr && cc == tc
          total_dist = cd
          break
        end
        if cr != 0
          nd = cd + @e_vert[cc][cr - 1]
          if li != 0
            nd -= {bonus_unvisited // (@c_vert[cc][cr - 1] + 1), @e_vert[cc][cr - 1]}.min
          end
          if nd < @sp_visited[cr - 1][cc]
            @sp_visited[cr - 1][cc] = nd
            @sp_dir[cr - 1][cc] = DIR_U
            @sp_q.add({-nd, cr - 1, cc})
          end
        end
        if cr != N - 1
          nd = cd + @e_vert[cc][cr]
          if li != 0
            nd -= {bonus_unvisited // (@c_vert[cc][cr] + 1), @e_vert[cc][cr]}.min
          end
          if nd < @sp_visited[cr + 1][cc]
            @sp_visited[cr + 1][cc] = nd
            @sp_dir[cr + 1][cc] = DIR_D
            @sp_q.add({-nd, cr + 1, cc})
          end
        end
        if cc != 0
          nd = cd + @e_horz[cr][cc - 1]
          if li != 0
            nd -= {bonus_unvisited // (@c_horz[cr][cc - 1] + 1), @e_horz[cr][cc - 1]}.min
          end
          if nd < @sp_visited[cr][cc - 1]
            @sp_visited[cr][cc - 1] = nd
            @sp_dir[cr][cc - 1] = DIR_L
            @sp_q.add({-nd, cr, cc - 1})
          end
        end
        if cc != N - 1
          nd = cd + @e_horz[cr][cc]
          if li != 0
            nd -= {bonus_unvisited // (@c_horz[cr][cc] + 1), @e_horz[cr][cc]}.min
          end
          if nd < @sp_visited[cr][cc + 1]
            @sp_visited[cr][cc + 1] = nd
            @sp_dir[cr][cc + 1] = DIR_R
            @sp_q.add({-nd, cr, cc + 1})
          end
        end
      end
      ave_cost = total_dist // ((sr - tr).abs + (sc - tc).abs)
      debugf("%d ave_cost:%d th_ave_cost:%d\n", @qi, ave_cost, th_ave_cost)
      break if ave_cost < th_ave_cost
    end
    ans = [] of Int32
    cr = tr
    cc = tc
    while cr != sr || cc != sc
      ans << @sp_dir[cr][cc]
      case ans[-1]
      when DIR_U
        @c_vert[cc][cr] += 1
        cr += 1
      when DIR_D
        @c_vert[cc][cr - 1] += 1
        cr -= 1
      when DIR_L
        @c_horz[cr][cc] += 1
        cc += 1
      when DIR_R
        @c_horz[cr][cc - 1] += 1
        cc -= 1
      end
    end
    return ans.reverse
  end

  def predict
    div = (ENV["repre_div"]? || "15").to_i
    ita = (ENV["repre_ita"]? || "8").to_i * 0.1
    rep = (ENV["repre_loop"]? || "50").to_i
    rep2 = (ENV["repre_loop2"]? || "60").to_i
    reset_freq = (ENV["reset_freq"]? || "100").to_i
    reset_until = (ENV["reset_until"]? || "900").to_i
    start_detect_x = (ENV["start_detect_x"]? || "500").to_i
    if @qi <= reset_until && @qi % reset_freq == 0
      N.times do |i|
        @e_horz[i].fill(5000)
        @e_vert[i].fill(5000)
        rep = rep2
      end
    end
    {% if !flag?(:local) %}
      elapsed = Time.utc.to_unix_ms - START_TIME
      if elapsed > 1950
        debug("timeout:#{@qi} #{elapsed}")
        rep = 0
      elsif elapsed > 1850
        debug("timeout:#{@qi} #{elapsed}")
        rep = 1
      end
    {% end %}
    dif_max = 1.05
    dif_min = 0.95

    @history.each do |h|
      h.skip_until = 0
      h.dif_hist.clear
      h.sum_ratio = 0.0
      h.hs.each do |p|
        h.sum_ratio += 1.0 / (@c_horz[p[0]][p[1]] + div)
      end
      h.vs.each do |p|
        h.sum_ratio += 1.0 / (@c_vert[p[0]][p[1]] + div)
      end
    end
    hs = @history.dup
    rep.times do |l|
      (@history.size - 1).times do |i|
        pos = @rnd.next_int(@history.size - i).to_i + i
        hs[i], hs[pos] = hs[pos], hs[i]
      end
      hs.each do |h|
        if h.skip_until > l
          next
        end
        sum = 0
        h.hs.each do |p|
          sum += @e_horz[p[0]][p[1]]
        end
        h.vs.each do |p|
          sum += @e_vert[p[0]][p[1]]
        end
        r = h.b / sum
        if r < dif_min
          diff = (h.b / dif_min - sum) * ita / h.sum_ratio
        elsif dif_max < r
          diff = (h.b / dif_max - sum) * ita / h.sum_ratio
        else
          if 0.99 < r && r < 1.01
            h.skip_until = l + 10
          elsif 0.97 < r && r < 1.03
            h.skip_until = l + 4
          elsif 0.96 < r && r < 1.04
            h.skip_until = l + 2
          end
          # h.dif_hist << {0, h.b / sum}
          # debugf("b:%d sum:%d diff:0\n", h.b, sum)
          next
        end
        # diff = (h.b - sum) * ita / sum_ratio
        # h.dif_hist << {diff.to_i, h.b / sum}
        # debugf("b:%d sum:%d diff:%d\n", h.b, sum, diff)

        h.hs.each do |p|
          @e_horz[p[0]][p[1]] += (diff * 1.0 / (@c_horz[p[0]][p[1]] + div) + 0.5).to_i
        end
        h.vs.each do |p|
          @e_vert[p[0]][p[1]] += (diff * 1.0 / (@c_vert[p[0]][p[1]] + div) + 0.5).to_i
        end
      end
      N.times do |i|
        (N - 1).times do |j|
          @e_horz[i][j] = {@e_horz[i][j], 1000}.max
          @e_horz[i][j] = {@e_horz[i][j], 9000}.min
          @e_vert[i][j] = {@e_vert[i][j], 1000}.max
          @e_vert[i][j] = {@e_vert[i][j], 9000}.min
        end
      end
      smoothing(3)
    end
    if @qi >= start_detect_x && @qi % 100 == 0 && !@is_m0
      N.times do |i|
        debug("detect_xh[#{i}]")
        @xh[i] = detect_x(@e_horz[i])
      end
      N.times do |i|
        debug("detect_xv[#{i}]")
        @xv[i] = detect_x(@e_vert[i])
      end
      cnt_x = @xh.count { |v| v != N } + @xv.count { |v| v != N }
      debug("cnt_x:#{cnt_x}")
      if @qi == start_detect_x && cnt_x < 9
        @is_m0 = true
        @xh.fill(N)
        @xv.fill(N)
      end
    end
    # if @qi % 10 == 0
    #   @history.each do |h|
    #     debug(h.dif_hist.join("\n"))
    #     debug("")
    #   end
    # end
  end

  def detect_x(row)
    detect_th = (ENV["detect_th"]? || "2400").to_i
    ave_pre = 0.0
    ave_suf = 0.0
    pre_sum = 0.0
    pre_sum2 = 0.0
    3.times do |i|
      ave_pre += row[i]
      ave_suf += row[N - 2 - i]
      pre_sum += row[i]
      pre_sum2 += row[i] * row[i]
    end
    ave_pre /= 3
    ave_suf /= 3
    if (ave_pre - ave_suf).abs < detect_th
      return N
    end
    total = 0.0
    total2 = 0.0
    row.each do |v|
      total += v
      total2 += v * v
    end
    min_var = 1e20
    min_i = N
    3.upto(N - 4) do |i|
      pre_sum += row[i]
      pre_sum2 += row[i] * row[i]
      suf_sum = total - pre_sum
      suf_sum2 = total2 - pre_sum2
      var_pre = pre_sum2 / (i + 1) - (pre_sum / (i + 1)) ** 2
      var_suf = suf_sum2 / (N - i - 2) - (suf_sum / (N - i - 2)) ** 2
      if var_pre + var_suf < min_var
        min_var = var_pre + var_suf
        min_i = i + 1
      end
    end
    debug("min_var:#{min_var} min_i:#{min_i}")
    return min_i
  end

  def postprocess
    div = (ENV["repre_div"]? || "15").to_i
    ita = @ita
    @history.reverse.each do |h|
      sum = 0
      sum_ratio = 0.0
      h.hs.each do |p|
        sum += @e_horz[p[0]][p[1]]
        sum_ratio += 1.0 / (@c_horz[p[0]][p[1]] + div)
      end
      h.vs.each do |p|
        sum += @e_vert[p[0]][p[1]]
        sum_ratio += 1.0 / (@c_vert[p[0]][p[1]] + div)
      end
      diff = (h.b - sum) * ita / sum_ratio
      # debugf("b:%d sum:%d diff:%d\n", h.b, sum, diff)
      h.hs.each do |p|
        @e_horz[p[0]][p[1]] += (diff * 1.0 / (@c_horz[p[0]][p[1]] + div) + 0.5).to_i
      end
      h.vs.each do |p|
        @e_vert[p[0]][p[1]] += (diff * 1.0 / (@c_vert[p[0]][p[1]] + div) + 0.5).to_i
      end
      ita = @ita2
    end
    N.times do |i|
      (N - 1).times do |j|
        @e_horz[i][j] = {@e_horz[i][j], 1000}.max
        @e_horz[i][j] = {@e_horz[i][j], 9000}.min
        @e_vert[i][j] = {@e_vert[i][j], 1000}.max
        @e_vert[i][j] = {@e_vert[i][j], 9000}.min
      end
    end
    len = (ENV["smo_l"]? || "3").to_i
    smoothing(len)
    # @e_horz.each do |row|
    #   debug(row.join(" "))
    # end
    # debug("")
    # @e_vert.each do |row|
    #   debug(row.join(" "))
    # end
  end

  def smoothing(len)
    mul = (ENV["smo_r"]? || "0.4").to_f
    sum = Array.new(N, 0)
    buf = Array.new(N - 1, 0)
    N.times do |i|
      (N - 1).times do |j|
        sum[j + 1] = sum[j] + @e_horz[i][j]
      end
      (N - 1).times do |j|
        left = j - len
        right = j + len
        if left < 0
          right += -left
          left = 0
        end
        if right >= N - 1
          left -= (right - N + 2)
          right = N - 2
        end
        if j < @xh[i] && @xh[i] <= right
          right = @xh[i] - 1
        elsif j >= @xh[i] && left < @xh[i]
          left = @xh[i]
        end
        ave = (sum[right + 1] - sum[left]) / (right - left + 1)
        buf[j] = @e_horz[i][j] + ((ave - @e_horz[i][j]) * mul).to_i
      end
      (N - 1).times do |j|
        @e_horz[i][j] = buf[j]
      end
    end
    N.times do |i|
      (N - 1).times do |j|
        sum[j + 1] = sum[j] + @e_vert[i][j]
      end
      (N - 1).times do |j|
        top = j - len
        bottom = j + len
        if top < 0
          bottom += -top
          top = 0
        end
        if bottom >= N - 1
          top -= (bottom - N + 2)
          bottom = N - 2
        end
        if j < @xv[i] && @xv[i] <= bottom
          bottom = @xv[i] - 1
        elsif j >= @xv[i] && top < @xv[i]
          top = @xv[i]
        end
        ave = (sum[bottom + 1] - sum[top]) / (bottom - top + 1)
        buf[j] = @e_vert[i][j] + ((ave - @e_vert[i][j]) * mul).to_i
      end
      (N - 1).times do |j|
        @e_vert[i][j] = buf[j]
      end
    end
  end
end
