#!/usr/bin/perl
use strict;

if($#ARGV < 0){
  print STDERR "One argument required (Dictionary lookup output file)\n";
  exit;
}

open my $dictFP, '>', $ARGV[0] or die("One argument required");
my %words;
my %tags;
my $word;
my $pos="";

while(<STDIN>){
  my @words = split /\s+/;
  my $buf;
  for my $element (@words){
    $pos = "";
#    $element = lc($element);
    if($element =~ m/(.+)\/(.+)/){
      $pos = $1;
      $word = $2;
    }else{
      $word = $element;
    }
      
    if(!exists($words{$word})){
      my $hashSize = keys %words;
      $words{$word} = $hashSize+1;
#      print STDERR "Size of words is: ".$hashSize."\n";
    }
    if(length($pos) > 0 && !exists($tags{$pos})){
        my $hashSize = keys %tags;
        $tags{$pos} = $hashSize + 1;
    }
    if(length($pos) > 0){
        $buf .= "$tags{$pos}/$words{$word} ";
    }else{
        $buf .= "$words{$word} ";
    }
  }
  print substr($buf, 0, length($buf)-1)."\n";
}

for my $key (sort keys %words){
  print $dictFP "$key $words{$key}\n";
}

