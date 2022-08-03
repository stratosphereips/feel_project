#awk -v inc='300' -f awk.tst conn.log 
{
    #print $1
    bucket = int($1/inc)
    # print int(bucket%inc) " " bucket*inc "-"  inc*(bucket+1)-1 " " NR 
    # system("mkdir -p out_dir/"(bucket%inc))
    # system("cp ssl.log.labeled out_dir/"(bucket%inc))
    # print $0  > "out_dir/"(bucket%inc)"/conn."( (inc*bucket) "-" (inc*(bucket+1)-1) ".log" )
    print $0  > "out_dir/conn."( (inc*bucket) "-" (inc*(bucket+1)-1) ".log" )
}
