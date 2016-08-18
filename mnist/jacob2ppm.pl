#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Std;

my %opts = (n=>0, c=>255.);
getopts('n:', \%opts);

my $d = 28;
my ($num, $coef) = ($opts{n}, $opts{c});

print "P3\n";
print "$d $d\n255\n";

while (<>) {
	chomp;
	my @t = split("\t");
	my $x = shift(@t);
	if ($x eq $num) {
		for (my $i = 0; $i < $d; ++$i) {
			my $z = $i * $d;
			for (my $j = 0; $j < $d; ++$j) {
				my $y = $t[$z + $j] * $coef;
				if ($y < 0) {
					$y = int(-$y + .499);
					$y = $y < 255? $y : 255;
					printf("   0   0 %3d", $y);
				} elsif ($y > 0) {
					$y = int($y + .499);
					$y = $y < 255? $y : 255;
					printf(" %3d   0   0", $y);
				} else {
					printf("   0   0   0");
				}
			}
			print "\n";
		}
	}
}
