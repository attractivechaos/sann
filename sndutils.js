var getopt = function(args, ostr) {
	var oli; // option letter list index
	if (typeof(getopt.place) == 'undefined')
		getopt.ind = 0, getopt.arg = null, getopt.place = -1;
	if (getopt.place == -1) { // update scanning pointer
		if (getopt.ind >= args.length || args[getopt.ind].charAt(getopt.place = 0) != '-') {
			getopt.place = -1;
			return null;
		}
		if (getopt.place + 1 < args[getopt.ind].length && args[getopt.ind].charAt(++getopt.place) == '-') { // found "--"
			++getopt.ind;
			getopt.place = -1;
			return null;
		}
	}
	var optopt = args[getopt.ind].charAt(getopt.place++); // character checked for validity
	if (optopt == ':' || (oli = ostr.indexOf(optopt)) < 0) {
		if (optopt == '-') return null; //  if the user didn't specify '-' as an option, assume it means null.
		if (getopt.place < 0) ++getopt.ind;
		return '?';
	}
	if (oli+1 >= ostr.length || ostr.charAt(++oli) != ':') { // don't need argument
		getopt.arg = null;
		if (getopt.place < 0 || getopt.place >= args[getopt.ind].length) ++getopt.ind, getopt.place = -1;
	} else { // need an argument
		if (getopt.place >= 0 && getopt.place < args[getopt.ind].length)
			getopt.arg = args[getopt.ind].substr(getopt.place);
		else if (args.length <= ++getopt.ind) { // no arg
			getopt.place = -1;
			if (ostr.length > 0 && ostr.charAt(0) == ':') return ':';
			return '?';
		} else getopt.arg = args[getopt.ind]; // white space
		getopt.place = -1;
		++getopt.ind;
	}
	return optopt;
}

function snd_rank(args)
{
	var c, zero_trun = false;
	while ((c = getopt(args, "0")) != null)
		if (c == '0') zero_trun = true;

	if (getopt.ind == args.length) {
		print("Usage: k8 sndutils.js rank [-0] <in.snd>");
		exit(1);
	}

	var buf = new Bytes();
	var file = new File(args[getopt.ind]);

	while (file.readline(buf) >= 0) {
		var t = buf.toString().split("\t");
		if (t[0].charAt(0) == '#') {
			print(t.join("\t"));
			continue;
		}
		var a = [];
		for (var i = 1; i < t.length; ++i) {
			var x = t[i] == "0"? 0.0 : parseFloat(t[i]);
			a.push([i, x, 0]);
		}
		a = a.sort(function(a,b) { return a[1]-b[1] });
		var i0 = 0;
		if (zero_trun) {
			for (i0 = 0; i0 < a.length; ++i0)
				if (a[i0][1] > 0.0) break;
				else a[i0][2] = 0;
		}
		var last = a[i0][1], n = 1;
		for (var i = i0 + 1; i <= a.length; ++i) {
			if (i == a.length || a[i][1] > last) {
				var y = (i - i0 - n + .5 * (n - 1)) / (a.length - i0 - 1);
				for (var j = i - n; j < i; ++j)
					a[j][2] = y.toFixed(6);
				if (i < a.length)
					n = 1, last = a[i][1];
			} else ++n;
		}
		a = a.sort(function(a,b) { return a[0]-b[0] });
		var o = [];
		for (var i = 0; i < a.length; ++i) o.push(a[i][2]);
		print(t[0], o.join("\t"));
	}

	file.close();
	buf.destroy();
}

function snd_selcol(args)
{
	if (args.length == 0) {
		print("Usage: k8 sndutils.js selcol <col.list> [in.snd]");
		exit(1);
	}

	var h = {}, file, buf = new Bytes();

	file = new File(args[0]);
	while (file.readline(buf) >= 0) {
		var t = buf.toString().split("\t");
		h[t[0]] = 1;
	}
	file.close();

	var a = [];
	file = args.length > 1? new File(args[1]) : new File();
	while (file.readline(buf) >= 0) {
		var t = buf.toString().split("\t");
		if (t[0].charAt(0) == '#') {
			a.push(0);
			for (var i = 1; i < t.length; ++i)
				if (h[t[i]]) a.push(i);
			var s = [];
			for (var i = 0; i < a.length; ++i)
				s.push(t[a[i]]);
			print(s.join("\t"));
			continue;
		}
		var s = [];
		for (var i = 0; i < a.length; ++i)
			s.push(t[a[i]]);
		print(s.join("\t"));
	}
	file.close();

	buf.destroy();
}

function snd_noise(args)
{
	var c, r = 0.3;
	while ((c = getopt(args, "r:")) != null)
		if (c == 'r') r = parseFloat(getopt.arg);
	if (getopt.ind == args.length) {
		print("Usage: k8 sndutils.js noise [-r 0.3] <in.snd>");
		exit(1);
	}

	var buf= new Bytes();
	var file = new File(args[getopt.ind]);
	while (file.readline(buf) >= 0) {
		var t = buf.toString().split("\t");
		if (t[0].charAt(0) != '#') {
			for (var i = 1; i < t.length; ++i)
				if (Math.random() < r) t[i] = 0.0;
		}
		print(t.join("\t"));
	}
	file.close();
	buf.destroy();
}

function snd_top(args)
{
	var c;
	while ((c = getopt(args, "")) != null);
	if (getopt.ind == args.length) {
		print("Usage: k8 sndutils.js top <in.snd>");
		exit(1);
	}

	var buf= new Bytes();
	var file = new File(args[getopt.ind]);
	var h;
	while (file.readline(buf) >= 0) {
		var t = buf.toString().split("\t");
		if (t[0].charAt(0) == '#') {
			h = t.slice(0);
		} else {
			var max = -1, max2 = -1, i1 = -1, i2 = -1;
			for (var i = 1; i < t.length; ++i) {
				var x = parseFloat(t[i]);
				if (x > max) max2 = max, i2 = i1, max = x, i1 = i;
				else if (x > max2) max2 = x, i2 = i;
			}
			print(t[0], max.toFixed(6), max2.toFixed(6), h[i1], h[i2]);
		}
	}
	file.close();
	buf.destroy();
}

if (arguments.length == 0) {
	print("Usage: k8 sndutils.js <command> <arguments>");
	print("Commands:");
	print("  rank      convert numbers to zero-truncated ranks");
	print("  selcol    select columns based on their names");
	print("  noise     randomly turn some values to zero");
	print("  top       top 2 classes");
}

var cmd = arguments.shift();
if (cmd == 'rank') snd_rank(arguments);
else if (cmd == 'selcol') snd_selcol(arguments);
else if (cmd == 'noise') snd_noise(arguments);
else if (cmd == 'top') snd_top(arguments);
else throw Error("ERROR: unknown command " + cmd);
