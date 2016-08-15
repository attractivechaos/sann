#!/usr/bin/perl

use strict;
use warnings;

while (<>) {
	if (/^#/) {
		print "#no:truth";
		for (my $i = 0; $i < 10; ++$i) {
			print "\t$i";
		}
		print "\n";
	} elsif (/^(\S+):(\d)/) {
		my @a = ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
		$a[$2] = 1;
		print "$1:$2\t", join("\t", @a), "\n";
	}
}
