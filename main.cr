START_TIME = Time.utc.to_unix_ms
TL         = 1800
N          =   30
INF        =  1e9
COUNTER    = Counter.new
DIR_CHARS  = "UDLR".chars
DIR_U      = 0
DIR_D      = 1
DIR_L      = 2
DIR_R      = 3
DR         = [-1, 1, 0, 0]
DC         = [0, 0, -1, 1]

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
  {% if flag?(:local) %}
    STDERR.puts({{msg}})
  {% end %}
end

macro debugf(format_string, *args)
  {% if flag?(:local) %}
    STDERR.printf({{format_string}}, {{*args}})
  {% end %}
end

macro chmin(ary, i, j, k, l, v)
  ary[i][j][k][k] = {ary[i][j][k][k], v}.min
end

class AtCoderJudge
  def query
    ps = read_line.split.map(&.to_i)
    return Tuple(Int32, Int32, Int32, Int32).from(ps)
  end

  def response(path)
    puts path.map { |d| DIR_CHARS[d] }.join
    return read_line.to_f
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
    @horz = generate_edge(d, m)
    @vert = generate_edge(d, m).transpose
    @sr = 0
    @sc = 0
    @tr = 0
    @tc = 0
    @qi = 0
    @input_file = File.new("out/#{@seed}.in.txt", "w")
    @output_file = File.new("out/#{@seed}.out.txt", "w")
    if f = @input_file
      N.times do |i|
        f << @horz[i].join(" ") << "\n"
      end
      (N - 1).times do |i|
        f << @vert[i].join(" ") << "\n"
      end
    end
  end

  private def generate_edge(d, m)
    ret = Array.new(N) { [0] * (N - 1) }
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
    end
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
    # debugf("qi:%d best:%d actual:%d\n", @qi, sd, dist)
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
    return (dist * e).round
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
  {% end %}
end

class Solver(Judge)
  def initialize(@judge : Judge, @timelimit : Int64)
    @rnd = XorShift.new(2u64)
  end

  def solve
    1000.times do
      sr, sc, tr, tc = @judge.query
      rev = sr > tr
      if rev
        sr, tr = tr, sr
        sc, tc = tc, sc
      end
      ans = [] of Int32
      (tr - sr).times do
        ans << DIR_D
      end
      if sc < tc
        ans += [DIR_R] * (tc - sc)
      else
        ans += [DIR_L] * (sc - tc)
      end
      if rev
        ans = ans.reverse.map { |v| v ^ 1 }
      end
      rough_dist = @judge.response(ans)
    end
  end
end
