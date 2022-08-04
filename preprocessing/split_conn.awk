# Run the script as follows:
# awk -v inc='3600' -v out_dir="out_dir" -f tst.awk conn.log.labeled 
{
    bucket = int($1/inc)
    start = inc*bucket
    # This one is ugly because it tries to create the dir for every line
    # Can't find a better way to do this atm
    system("mkdir -p out_dir/"(start))
    print $0  >> out_dir"/"start"/conn."( (inc*bucket) "-" (inc*(bucket+1)-1) ".log" )
}
