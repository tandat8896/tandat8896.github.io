// Standalone diagram source. Prints SVG or native draw.io XML; never writes files.
// Usage: node designs/architecture/render.mjs <svg|drawio> <id|all>
import { readFileSync } from 'node:fs';
const C = { ink:'#232f3e', muted:'#526477', line:'#64748b', compute:'#c46b08', database:'#3569a8', network:'#7654a3', storage:'#398153', control:'#977316' };
const fills = { compute:'#fff8ee', database:'#f2f7fc', network:'#f8f5fc', storage:'#f3faf5', control:'#fffbef', ink:'#f8fafc' };
const diagrams = [];
function graph(id,title,subtitle,footnote,height=1000) {
  const d={id,title,subtitle,footnote,width:1600,height,zones:[],nodes:[],edges:[],notes:[]}; diagrams.push(d);return d;
}
function zone(d,id,label,x,y,w,h,tone='network',parent=null) { d.zones.push({id,label,x,y,w,h,tone,parent}); }
function node(d,id,label,detail,x,y,tone='compute',icon='server',parent=null,w=180,h=100) { d.nodes.push({id,label,detail,x,y,w,h,tone,icon,parent}); }
function edge(d,from,to,points,label='',tone='ink',kind='flow',lx=null,ly=null) { d.edges.push({from,to,points,label,tone,kind,lx,ly}); }
function note(d,text,x,y,w=400,tone='muted') {d.notes.push({text,x,y,w,tone});}

// 01 — physical connectivity. Lines represent network attachment, not packet direction.
{
const d=graph('network-topology','Data engineering homelab','NETWORK TOPOLOGY  /  NixOS · KVM/libvirt · segmented virtual networks','One physical NixOS host. Segmentation and VM-level HA do not provide independent-site disaster recovery.',1080);
zone(d,'host','ON-PREMISES  /  NixOS hypervisor',40,140,1520,800,'ink');
zone(d,'bdpub','COMPUTE NETWORK  ·  10.10.20.0/24',70,210,530,270,'compute','host');
zone(d,'racpub','DATABASE PUBLIC  ·  10.10.50.0/24',980,210,550,400,'database','host');
zone(d,'bdpriv','HDFS STORAGE NETWORK  ·  10.10.30.0/24',70,660,530,230,'storage','host');
zone(d,'racpriv','RAC PRIVATE  ·  10.10.10.0/24',980,660,550,230,'network','host');
node(d,'bd1','datanode','.10  /  Spark master',90,280,'compute','server','bdpub',145);
node(d,'bd2','workernode1','.11  /  master + worker',260,280,'compute','server','bdpub',145);
node(d,'bd3','workernode2','.12  /  Spark worker',430,280,'compute','server','bdpub',145);
node(d,'fw','Host L3/L4 firewall','Forwarding policy',700,340,'network','firewall','host',200,100);
node(d,'scan','SCAN listener','.30:1521',1000,280,'network','router','racpub',150);
node(d,'rac1','rac1 / racdb1','.11  ·  VIP .21',1170,280,'database','server','racpub',150);
node(d,'rac2','rac2 / racdb2','.12  ·  VIP .22',1350,280,'database','server','racpub',150);
node(d,'stdby','stdby1 / racdb_s','.40  /  physical standby',1160,470,'database','database','racpub',190);
node(d,'hdfs','Dual-NIC cluster','Storage NICs: .10 / .11 / .12',190,730,'storage','switch','bdpriv',290);
node(d,'cache','Cache Fusion interconnect','Private NICs: .11 / .12 · MTU 9000',1090,730,'network','switch','racpriv',340);
edge(d,'bd1','fw',[[163,380],[163,430],[660,430],[660,390],[700,390]],'','compute','link');
edge(d,'bd2','fw',[[333,380],[333,430],[660,430],[660,390],[700,390]],'','compute','link');
edge(d,'bd3','fw',[[503,380],[503,430],[660,430],[660,390],[700,390]],'','compute','link');
edge(d,'fw','scan',[[900,390],[940,390],[940,330],[1000,330]],'1. Connect :1521','network','flow',915,304);
edge(d,'fw','rac1',[[800,340],[800,190],[1250,190],[1250,280]],'TCP 1521 · SCAN + VIPs only','network','flow',800,178);
edge(d,'fw','rac2',[[800,340],[800,190],[1425,190],[1425,280]],'','network','flow');
edge(d,'rac1','stdby',[[1250,380],[1250,470]],'Thread 1 redo','database','flow',1200,427);
edge(d,'rac2','stdby',[[1425,380],[1425,520],[1350,520]],'Thread 2 redo','database','flow',1425,460);
edge(d,'bd1','hdfs',[[190,380],[190,610],[335,610],[335,730]],'','storage','link');
edge(d,'bd2','hdfs',[[360,380],[360,610],[335,610],[335,730]],'Second NICs · all three VMs','storage','link',335,586);
edge(d,'bd3','hdfs',[[530,380],[530,610],[335,610],[335,730]],'','storage','link');
edge(d,'rac1','cache',[[1320,330],[1335,330],[1335,440],[1480,440],[1480,630],[1260,630],[1260,730]],'','network','link');
edge(d,'rac2','cache',[[1460,380],[1505,380],[1505,630],[1260,630],[1260,730]],'Private interconnect','network','link',1360,636);
note(d,'2. SCAN returns a VIP; client reconnects',650,475,310,'network');
note(d,'ALLOW  1521 → SCAN .30, VIP .21 / .22',650,510,310,'network');
note(d,'DROP  compute → DB SSH / direct node IP',650,535,310);
note(d,'No cross-zone forwarding for storage',650,570,310);
note(d,'virbr-bd-pub  /  gateway 10.10.20.1',95,458,450);
note(d,'virbr-rac-pub  /  gateway 10.10.50.1',1005,588,490);
note(d,'virbr-bd-priv',95,868,450);
note(d,'virbr-racpriv',1005,868,490);
}

// 02 — connection, shared storage, redo transport and observer are separate paths.
{
const d=graph('oracle-rac','Oracle RAC & Data Guard','DATABASE ARCHITECTURE  /  two active instances · shared ASM · physical standby','Topology from lab runbooks; no measured availability or zero-data-loss SLA is implied.',1060);
zone(d,'host','ON-PREMISES  /  NixOS + KVM',40,260,1520,660,'ink');
zone(d,'primary','PRIMARY CLUSTER  /  racdb',80,330,880,540,'database','host');
zone(d,'private','PRIVATE INTERCONNECT  /  10.10.10.0/24',310,530,430,110,'network','primary');
zone(d,'standbyzone','STANDBY VM',1090,330,420,370,'database','host');
node(d,'app','Application / Spark','JDBC client',80,140,'compute','client',null,230);
node(d,'scan','SCAN listener','10.10.50.30:1521',390,140,'network','router',null,240);
node(d,'one','rac1  /  racdb1','10.10.50.11 · VIP .21',170,410,'database','server','primary',260);
node(d,'two','rac2  /  racdb2','10.10.50.12 · VIP .22',620,410,'database','server','primary',260);
node(d,'asm','Shared Oracle ASM','+OCRVOTE 10 GB · +DATA 30 GB · +FRA 30 GB',230,720,'storage','storage','primary',560,100);
node(d,'standby','stdby1  /  racdb_s','10.10.50.40 · redo apply (MRP)',1160,460,'database','database','standbyzone',280);
node(d,'observer','Broker / FSFO observer','Monitor primary + standby',1120,140,'control','monitor',null,340);
edge(d,'app','scan',[[310,170],[390,170]],'1. Connect','ink','flow',350,153);
edge(d,'scan','app',[[390,215],[310,215]],'2. Redirect','network','control',350,240);
// Enter the host below its heading, keeping both boundary labels free of connectors.
edge(d,'app','one',[[80,175],[20,175],[20,370],[300,370],[300,410]],'3. Session → VIP .21:1521','network','flow',280,370);
edge(d,'app','two',[[80,220],[35,220],[35,305],[830,305],[830,410]],'3. Session → VIP .22:1521','network','flow',710,305);
edge(d,'one','two',[[300,510],[300,595],[750,595],[750,510]],'Cache Fusion · MTU 9000','network','duplex',525,595);
edge(d,'one','asm',[[240,510],[240,680],[400,680],[400,720]],'','storage','duplex');
edge(d,'two','asm',[[810,510],[810,680],[620,680],[620,720]],'Read / write shared blocks','storage','duplex',520,680);
edge(d,'one','standby',[[430,450],[470,450],[470,390],[970,390],[970,490],[1160,490]],'Redo · thread 1','database','flow',700,390);
edge(d,'two','standby',[[880,480],[1030,480],[1030,540],[1160,540]],'Redo · thread 2','database','flow',1075,540);
edge(d,'observer','standby',[[1300,240],[1300,460]],'Health / role control','control','control-duplex',1300,297);
edge(d,'observer','two',[[1120,190],[1030,190],[1030,365],[750,365],[750,410]],'','control','control-duplex');
note(d,'SCAN redirects the client; SQL sessions then use a VIP listener directly.',1115,625,355);
note(d,'Both RAC nodes open the same database.',110,850,670);
}

// 03 — host columns explain co-location; logical lanes are not additional machines.
{
const d=graph('hadoop-spark','Hadoop & Spark high availability','CLUSTER TOPOLOGY  /  resource allocation ≠ task execution ≠ HDFS data path','Logical service flows; HA dependencies below have one member on each of the three VMs. Representative HDFS client paths shown.',1390);
zone(d,'cluster','ON-PREMISES  /  three dual-NIC VMs',50,140,1500,1110,'ink');
zone(d,'vm1','datanode  ·  .10',90,230,430,770,'compute','cluster');
zone(d,'vm2','workernode1  ·  .11',580,230,430,770,'compute','cluster');
zone(d,'vm3','workernode2  ·  .12',1070,230,430,770,'compute','cluster');
zone(d,'ha','HA DEPENDENCIES  /  logical groups across .10, .11, .12',90,1050,1410,170,'control','cluster');
node(d,'driver','Spark driver','Client mode · on datanode',150,290,'compute','client','vm1',310);
node(d,'spark1','Spark master','Active role · :7077',160,480,'compute','compute','vm1',290);
node(d,'spark2','Spark master','Standby role · :7077',630,290,'compute','compute','vm2',320);
node(d,'worker1','Worker + executors','workernode1',640,480,'compute','server','vm2',300);
node(d,'worker2','Worker + executors','workernode2',1120,480,'compute','server','vm3',310);
node(d,'nn1','NameNode nn1 + ZKFC','Active role',160,720,'storage','database','vm1',290);
node(d,'nn2','nn2 + ZKFC','Standby role',600,720,'storage','database','vm2',180);
node(d,'dn1','DataNode','HDFS blocks',810,720,'storage','storage','vm2',180);
node(d,'dn2','DataNode','HDFS blocks',1130,720,'storage','storage','vm3',290);
node(d,'zk','ZooKeeper ensemble','3 members · election / recovery · :2181',160,1100,'control','cluster','ha',430);
node(d,'jn','JournalNode quorum','3 members · shared edits · :8485',910,1100,'storage','log','ha',430);
edge(d,'driver','spark1',[[305,390],[305,480]],'Request resources','compute','flow',305,437);
edge(d,'driver','spark2',[[460,330],[630,330]],'HA fallback','control','control',545,313);
edge(d,'spark1','worker1',[[450,520],[640,520]],'Launch executors','compute','flow',545,500);
edge(d,'spark1','worker2',[[450,550],[500,550],[500,665],[1275,665],[1275,580]],'Launch executors','compute','flow',1080,665);
edge(d,'driver','worker1',[[460,360],[540,360],[540,435],[790,435],[790,480]],'Tasks / results · RPC','compute','duplex',785,435);
edge(d,'driver','worker2',[[305,290],[305,195],[1040,195],[1040,425],[1275,425],[1275,480]],'Tasks / results · RPC','compute','duplex',750,195);
edge(d,'worker1','worker2',[[940,505],[1120,505]],'Shuffle blocks','compute','duplex',1030,480);
edge(d,'worker1','nn1',[[640,555],[550,555],[550,690],[400,690],[400,720]],'NameNode metadata','control','control-duplex',400,690);
edge(d,'worker1','nn2',[[715,580],[715,720]],'HA fallback','control','control',715,625);
edge(d,'worker1','dn1',[[900,580],[900,720]],'Read / write blocks','storage','duplex',900,625);
edge(d,'worker2','dn2',[[1390,580],[1390,720]],'Read / write blocks','storage','duplex',1370,625);
edge(d,'dn1','dn2',[[990,770],[1130,770]],'Block replication','storage','duplex',1060,752);
edge(d,'nn1','jn',[[305,820],[305,960],[1050,960],[1050,1100]],'Active writes shared edits','storage','flow',800,960);
edge(d,'jn','nn2',[[1150,1100],[1150,1020],[690,1020],[690,820]],'Standby tails shared edits','storage','flow',885,1020);
edge(d,'nn1','zk',[[180,820],[180,1100]],'ZKFC election','control','control-duplex',235,905);
edge(d,'nn2','zk',[[620,820],[620,990],[450,990],[450,1100]],'ZKFC','control','control-duplex',555,990);
edge(d,'spark1','zk',[[160,525],[125,525],[125,1040],[350,1040],[350,1100]],'Master recovery','control','control-duplex',245,1040);
edge(d,'spark2','zk',[[630,305],[560,305],[560,930],[540,930],[540,1100]],'','control','control-duplex');
note(d,'Compute 10.10.20.0/24 · HDFS storage 10.10.30.0/24',95,1238,1370,'muted');
note(d,'Task payloads go driver ↔ executors.',1090,320,390);
note(d,'HDFS blocks bypass the NameNodes.',1090,355,390);
}

// 04 — schema metadata is a side-channel, never an in-line event hop.
{
const d=graph('cdc-iceberg','CDC to the Iceberg lakehouse','IN PROGRESS / CHƯA HOÀN THIỆN  ·  PostgreSQL → Debezium → Kafka → Spark → Iceberg','Avro v4: batch Bronze + partial Silver. TODO: Silver → Supabase, end-to-end streaming. No Gold layer implemented.',1060);
zone(d,'local','LOCAL MIGRATION LAB  /  process boundaries, not separate cloud subnets',40,180,1520,730,'ink');
zone(d,'source','SOURCE',70,400,250,380,'database','local');
zone(d,'events','CAPTURE & EVENT TRANSPORT',360,280,530,500,'compute','local');
zone(d,'lake','PROCESSING & LAKEHOUSE',930,280,600,590,'storage','local');
note(d,'CHƯA HOÀN THIỆN  /  Silver → Supabase: TODO · Gold: chưa có · Streaming end-to-end: chưa có',80,242,1440,'compute');
node(d,'pg','PostgreSQL','pgoutput · :5432',90,510,'database','database','source',210);
node(d,'deb','Debezium Server','Avro → raw byte producer',380,510,'compute','compute','events',210);
node(d,'kafka','Apache Kafka','CDC topic · :9092',650,510,'compute','queue','events',210);
node(d,'schema','Confluent Schema Registry','Schema ID · BACKWARD · :8081',430,340,'control','schema','events',400);
node(d,'spark','Apache Spark','Avro decode · batch read',960,510,'compute','compute','lake',240);
node(d,'bronze','Iceberg Bronze','cdc_bronze_v4',1250,510,'storage','storage','lake',250);
node(d,'silver','Iceberg Silver','customers_scd2',1250,710,'storage','storage','lake',250);
edge(d,'pg','deb',[[300,560],[380,560]],'WAL','database','flow',340,535);
edge(d,'deb','kafka',[[590,560],[650,560]],'Avro','compute','flow',620,535);
edge(d,'kafka','spark',[[860,560],[960,560]],'Batch read','compute','flow',910,535);
edge(d,'spark','bronze',[[1200,560],[1250,560]],'Write','storage','flow',1225,535);
edge(d,'deb','schema',[[480,510],[480,440]],'Register','control','control',480,472);
edge(d,'spark','schema',[[1080,510],[1080,390],[830,390]],'GET schema ID / schema response','control','control-duplex',1110,368);
edge(d,'bronze','silver',[[1375,610],[1375,710]],'12_scd_type2.scala','storage','flow',1375,659);
note(d,'Initial snapshot + WAL changes',90,665,210);
note(d,'Event envelope: before · after · op · ts_ms',380,680,470);
note(d,'Registry stores schemas, not the event stream.',380,720,470);
note(d,'Local warehouse · Iceberg metadata + Parquet',970,846,510);
note(d,'Script 10: spark.read + createOrReplace',950,660,285);
note(d,'Silver exercise omits delete events;',950,745,285);
note(d,'history semantics need more work.',950,775,285);
}

// 05 — a process/module topology, not an invented network deployment.
{
const d=graph('tinydb','tinydb · inside a storage engine','SOFTWARE TOPOLOGY  /  Rust · single process · direct file I/O','WAL / Transaction is a separate API and is not wired into the current SQL REPL. No buffer pool, MVCC or full ARIES recovery.',1110);
zone(d,'process','RUST PROCESS  /  SQL path and separate transaction API',40,160,1520,650,'ink');
zone(d,'query','QUERY EXECUTION',70,240,280,410,'compute','process');
zone(d,'tablezone','TABLE  /  heap + in-memory index',410,240,730,520,'database','process');
zone(d,'txnzone','TRANSACTION API  /  wal.rs',1200,240,330,520,'control','process');
zone(d,'disk','LOCAL FILESYSTEM',410,830,1120,140,'storage');
node(d,'repl','SQL REPL','main.rs',100,310,'compute','client','query',220);
node(d,'parser','Lexer → Parser → Executor','INSERT · SELECT WHERE id',90,500,'compute','compute','query',240);
node(d,'table','Table','insert / get / open',470,310,'database','table','tablezone',240);
node(d,'index','B+Tree index','row ID → (page, slot)',840,310,'database','tree','tablezone',260);
node(d,'heap','Heap + 8 KiB pages','Row bytes · slotted pages',560,580,'storage','storage','tablezone',430);
node(d,'txn','Transaction','Pending rows in memory',1230,310,'control','compute','txnzone',270);
node(d,'wal','WAL append','flush + sync_all',1230,580,'control','log','txnzone',270);
node(d,'datafile','Heap data file','Table::open() rebuilds index',540,860,'storage','file','disk',450,100);
node(d,'walfile','Append-only WAL file','No crash replay implementation',1200,860,'storage','file','disk',300,100);
edge(d,'repl','parser',[[210,410],[210,500]],'SQL','compute','flow',210,454);
edge(d,'parser','table',[[330,550],[375,550],[375,360],[470,360]],'Execute AST','compute','flow',420,440);
edge(d,'table','index',[[710,340],[840,340]],'get(id) / insert(id, tid)','database','flow',780,305);
edge(d,'index','table',[[840,390],[710,390]],'Return tuple location','database','flow',780,438);
edge(d,'table','heap',[[590,410],[590,580]],'insert(bytes) / get(tid)','storage','duplex',590,483);
edge(d,'heap','datafile',[[775,680],[775,860]],'Write-through pages','storage','flow',775,800);
edge(d,'datafile','heap',[[940,860],[1040,860],[1040,630],[990,630]],'open(): read file into RAM','storage','flow',1010,725);
edge(d,'txn','wal',[[1365,410],[1365,580]],'1. Append + fsync','control','flow',1365,483);
edge(d,'wal','walfile',[[1365,680],[1365,860]],'Log bytes','storage','flow',1365,800);
edge(d,'txn','table',[[1230,350],[1170,350],[1170,210],[590,210],[590,310]],'2. Apply rows after WAL writes','control','flow',890,208);
note(d,'Rollback discards pending rows.',1220,723,290);
note(d,'The REPL currently creates a table on startup.',80,705,290);
}

// 06 — per-instance memory and process flow; parameter values are saved snapshots.
{
const d=graph('oracle-memory','Oracle · memory, processes & I/O','INSTANCE ARCHITECTURE  /  SGA · PGA · LGWR · DBWn · CKPT','Saved pre-tuning parameters, not live configuration. Each RAC instance has its own SGA and background processes.',1320);
zone(d,'instance','ONE ORACLE INSTANCE  /  pattern repeated for racdb1 and racdb2',40,290,1520,600,'ink');
zone(d,'foreground','FOREGROUND PROCESS',60,350,360,370,'compute','instance');
zone(d,'sga','SGA  /  shared within this instance',480,350,1040,290,'database','instance');
zone(d,'background','BACKGROUND PROCESSES',480,690,1040,170,'control','instance');
zone(d,'files','DATABASE STORAGE  /  ASM-managed files',60,940,1480,190,'storage');
node(d,'client','Application / SQL client','Dedicated connection',80,150,'compute','client',null,250);
node(d,'server','Server process','Parse · execute · fetch',80,400,'compute','compute','foreground',300);
node(d,'pga','PGA / SQL work areas','Private per process · sort / hash',80,590,'compute','compute','foreground',300);
node(d,'shared','Shared pool','SQL / PL/SQL · dictionary cache',520,430,'database','schema','sga',240);
node(d,'cache','Database buffer cache','Cached blocks · dirty buffers',840,430,'database','storage','sga',270);
node(d,'redo','Redo log buffer','Change records in memory',1230,430,'database','log','sga',240);
node(d,'ckpt','CKPT','Checkpoint coordination',520,740,'control','monitor','background',240);
node(d,'dbwn','DBWn','Database writer',840,740,'control','compute','background',270);
node(d,'lgwr','LGWR','Log writer',1230,740,'control','log','background',240);
node(d,'temp','TEMP files','Work-area spill when needed',90,990,'storage','file','files',260);
node(d,'control','Control files','Checkpoint metadata',490,990,'storage','file','files',260);
node(d,'data','Datafiles','Persisted database blocks',840,990,'storage','database','files',270);
node(d,'online','Online redo logs','Durable redo · instance thread',1230,990,'storage','log','files',240);
edge(d,'client','server',[[80,200],[20,200],[20,380],[205,380],[205,400]],'SQL / COMMIT','compute','flow',160,382);
edge(d,'server','pga',[[230,500],[230,590]],'Work-area use','compute','duplex',230,550);
edge(d,'server','shared',[[380,435],[450,435],[450,470],[520,470]],'Parse / plans','control','control-duplex',452,413);
edge(d,'server','cache',[[380,460],[435,460],[435,565],[975,565],[975,530]],'Read / modify blocks','database','duplex',885,565);
edge(d,'server','redo',[[380,485],[445,485],[445,615],[1355,615],[1355,530]],'Generate redo','database','flow',1260,615);
edge(d,'cache','dbwn',[[975,530],[975,740]],'Dirty buffers','storage','flow',975,671);
edge(d,'redo','lgwr',[[1355,530],[1355,740]],'Flush redo','database','flow',1355,671);
edge(d,'dbwn','data',[[975,840],[975,990]],'Write dirty blocks','storage','flow',975,916);
edge(d,'lgwr','online',[[1355,840],[1355,990]],'Write + durable flush','storage','flow',1355,916);
edge(d,'lgwr','server',[[1470,790],[1545,790],[1545,320],[450,320],[450,385],[290,385],[290,400]],'Commit ACK after redo flush · default WAIT semantics','control','control',970,320);
edge(d,'data','cache',[[840,1040],[795,1040],[795,480],[840,480]],'Cache-miss read','storage','flow',780,895);
edge(d,'pga','temp',[[220,690],[220,900],[460,900],[460,980],[220,980],[220,990]],'Spill / read work files','storage','duplex',220,860);
edge(d,'ckpt','dbwn',[[760,790],[840,790]],'Signal','control','control',800,773);
edge(d,'ckpt','control',[[620,840],[620,990]],'Checkpoint metadata','control','control',620,916);
edge(d,'ckpt','data',[[700,840],[700,875],[1140,875],[1140,1040],[1110,1040]],'Update datafile headers','control','control',1165,964);
note(d,'COMMIT waits for redo durability; dirty data blocks are written independently by DBWn.',80,1175,1410);
note(d,'Saved snapshot: SGA_TARGET 1.5 GiB · MEMORY_TARGET 0 (ASMM) · PGA target 512 MiB / limit 2 GiB.',80,1205,1410);
}

// Public portfolio additions: implemented boundaries, without unverified performance claims.
{
const d=graph('medallion-nix','Medallion data platform on Nix','LOCAL ETL LAB  /  Kafka · Flink · Spark · NDJSON → Parquet','Local lab implementation. PostgreSQL loaders exist; README leaves the load run and Airflow orchestration pending.',1120);
zone(d,'lab','NIX FLAKES ENVIRONMENT  /  local services and filesystem',40,160,1520,800,'ink');
zone(d,'ingest','STREAM INGESTION',80,250,430,650,'compute','lab');
zone(d,'lake','MEDALLION FILE LAYERS',580,250,420,650,'storage','lab');
zone(d,'compute','BATCH PROCESSING',1080,250,430,400,'compute','lab');
zone(d,'serving','SERVING  /  loader code, execution pending',1080,700,430,200,'database','lab');
node(d,'sources','Sample event sources','Five Kafka topics',125,330,'compute','client','ingest',330);
node(d,'kafka','Apache Kafka','Event transport',125,520,'compute','queue','ingest',330);
node(d,'flink','Apache Flink','Streaming ingest · FileSink',125,730,'compute','compute','ingest',330);
node(d,'bronze','Bronze','Raw NDJSON · source / dt / hour',625,330,'storage','file','lake',330);
node(d,'silver','Silver','Domain tables · Parquet / Snappy',625,520,'storage','file','lake',330);
node(d,'gold','Gold','Business KPI tables · Parquet',625,730,'storage','file','lake',330);
node(d,'spark','Apache Spark','Batch transforms · dedup / aggregate',1125,380,'compute','compute','compute',330);
node(d,'loader','Spark JDBC loaders','Append into KPI tables',1125,530,'compute','compute','compute',330);
node(d,'pg','PostgreSQL','etl_analytics · KPI schema',1125,760,'database','database','serving',330);
edge(d,'sources','kafka',[[290,430],[290,520]],'Events','compute','flow',290,477);
edge(d,'kafka','flink',[[290,620],[290,730]],'Consume','compute','flow',290,677);
edge(d,'flink','bronze',[[455,780],[545,780],[545,380],[625,380]],'NDJSON','storage','flow',548,447);
edge(d,'bronze','silver',[[790,430],[790,520]],'Spark · clean / dedup','storage','flow',790,477);
edge(d,'silver','gold',[[790,620],[790,730]],'Spark · aggregate','storage','flow',790,677);
edge(d,'spark','silver',[[1125,430],[1040,430],[1040,570],[955,570]],'Batch job','control','control',1040,510);
edge(d,'gold','loader',[[955,780],[1030,780],[1030,580],[1125,580]],'Read Gold','storage','flow',1030,678);
edge(d,'loader','pg',[[1455,580],[1530,580],[1530,810],[1455,810]],'Load: TODO','control','control',1510,675);
note(d,'Gold: daily revenue · user retention · inventory turnover.',625,855,360);
note(d,'IoT KPI is a placeholder; no production-readiness claim.',80,1005,1400);
note(d,'Batch drop-zone and additional infra are detailed in the original project drawing.',80,984,1400);
}
{
const d=graph('solar-pv','Solar PV · data, analytics & forecasting','GRADUATION PROJECT  /  The Outliers · Ngô Tấn Đạt: Data & AI Lead','Code-backed architecture. Optuna is a separate action; approved parameters are promoted manually.',1460);
zone(d,'ingest','DATA ENGINEERING  /  Python ETL',40,170,390,880,'compute');
zone(d,'warehouse','POSTGRESQL / SUPABASE  /  logical schemas',480,170,400,880,'database');
zone(d,'serving','ANALYTICS & APPLICATIONS',940,170,620,1120,'storage');
zone(d,'tuning','EXPERIMENTS  /  standalone tuning · not in the standard run',40,1090,840,200,'control');
node(d,'source','PV + weather data','CSV / archive weather data',85,280,'compute','file','ingest',300);
node(d,'extract','Extract / load','Object storage → staging',85,490,'compute','compute','ingest',300);
node(d,'transform','GMM–IF outlier detection','Imputation → GMM ∧ IF + guards',85,710,'compute','compute','ingest',300);
node(d,'staging','Staging / buffers','Raw records and transformed buffers',515,490,'database','database','warehouse',330);
node(d,'dwh','Galaxy warehouse','Two facts · five dimensions',515,710,'database','database','warehouse',330);
node(d,'bi','BI mart','Hourly measures / materialized views',970,320,'storage','table','serving',270);
node(d,'tableau','Tableau','Performance dashboards',1280,320,'storage','monitor','serving',250);
node(d,'ml','ML mart','Parquet feature data',970,580,'storage','file','serving',270);
node(d,'train','LightGBM forecasting','Train · evaluate · saved model',970,800,'compute','compute','serving',270);
node(d,'app','Streamlit','Forecast / What-if UI',1280,800,'compute','monitor','serving',250);
node(d,'optuna','Optuna · HPO','TPE search · CV folds · trial params',85,1140,'control','compute','tuning',300);
node(d,'approved','Approved parameters','best_params.json · reviewed / locked',515,1140,'control','file','tuning',330);
node(d,'shap','Explainable AI · SHAP','Global importance · local explanation',970,1140,'compute','compute','serving',270);
edge(d,'source','extract',[[235,380],[235,490]],'Extract','compute','flow',235,445);
edge(d,'extract','staging',[[385,540],[515,540]],'Load raw','database','flow',450,523);
edge(d,'staging','transform',[[515,565],[460,565],[460,760],[385,760]],'Transform','compute','flow',453,635);
edge(d,'transform','dwh',[[385,790],[460,790],[460,850],[680,850],[680,810]],'Load clean data','database','flow',640,850);
edge(d,'dwh','bi',[[845,740],[915,740],[915,370],[970,370]],'BI build','database','flow',914,480);
edge(d,'dwh','ml',[[845,770],[930,770],[930,630],[970,630]],'Features','database','flow',930,705);
edge(d,'bi','tableau',[[1240,370],[1280,370]],'Query','storage','flow',1260,350);
edge(d,'ml','train',[[1105,680],[1105,800]],'Train / evaluate','compute','flow',1105,745);
edge(d,'train','app',[[1240,850],[1260,850],[1260,960],[1410,960],[1410,900]],'Model / output artifacts','compute','flow',1340,960);
edge(d,'bi','app',[[1105,420],[1105,480],[1410,480],[1410,800]],'What-if inputs','storage','flow',1410,670);
edge(d,'optuna','approved',[[385,1190],[515,1190]],'Manual review','control','control',450,1168);
edge(d,'approved','train',[[845,1190],[900,1190],[900,920],[995,920],[995,900]],'Read approved params','control','control',900,1005);
edge(d,'train','shap',[[1105,900],[1105,1140]],'Model + feature data','compute','flow',1105,1050);
edge(d,'shap','app',[[1240,1190],[1490,1190],[1490,900]],'Explanations','compute','flow',1390,1190);
note(d,'GMM = Gaussian Mixture Model. IF = Isolation Forest. Consensus and physical guards produce anomaly flags.',80,1320,1430);
note(d,'Optuna trial output does not overwrite locked parameters. Screenshots are report snapshots, not live telemetry.',80,1343,1430);
}

function esc(s) { return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c])); }
const icons = {
 server:'<rect x="17" y="9" width="30" height="46" rx="2"/><path d="M17 24h30M17 39h30M23 17h12M23 32h12M23 47h12"/><circle cx="41" cy="17" r="1"/><circle cx="41" cy="32" r="1"/>',
 database:'<ellipse cx="32" cy="14" rx="21" ry="8"/><path d="M11 14v36c0 11 42 11 42 0V14M11 31c0 11 42 11 42 0M11 44c0 11 42 11 42 0"/>',
 storage:'<ellipse cx="32" cy="14" rx="22" ry="8"/><path d="M10 14v37c0 10 44 10 44 0V14M10 33c0 10 44 10 44 0"/><path d="M23 46h18"/>',
 router:'<path d="M12 23h40v23H12zM22 23v-9h20v9M22 46v9M42 46v9M25 31l-5 4 5 4m14-8 5 4-5 4M20 35h24"/>',
 switch:'<rect x="7" y="23" width="50" height="25" rx="2"/><path d="M14 31h7v8h-7zm14 0h7v8h-7zm14 0h7v8h-7zM20 23V10m24 13V10M14 10h12m12 0h12"/>',
 firewall:'<path d="M12 13h40v39H12zM12 26h40M12 39h40M25 13v13M40 13v13M20 26v13M36 26v13M25 39v13M40 39v13"/>',
 client:'<rect x="9" y="10" width="46" height="32" rx="2"/><path d="M24 42v11m16-11v11M17 54h30M24 20l-6 6 6 6m16-12 6 6-6 6"/>',
 compute:'<rect x="17" y="17" width="30" height="30" rx="2"/><path d="M25 17V8m14 9V8M25 47v9m14-9v9M17 25H8m9 14H8m39-14h9m-9 14h9M26 26h12v12H26z"/>',
 queue:'<path d="M10 12h44v40H10zM10 25h44M10 39h44M19 18h8m-8 14h16m-16 14h24"/>',
 schema:'<path d="M17 8h23l10 11v38H17zM40 8v13h10M26 29l-5 6 5 6m15-12 5 6-5 6M35 28l-4 15"/>',
 cluster:'<circle cx="32" cy="14" r="8"/><circle cx="13" cy="48" r="8"/><circle cx="51" cy="48" r="8"/><path d="M28 21L17 41m19-20 11 20M21 48h22"/>',
 monitor:'<rect x="8" y="12" width="48" height="35" rx="2"/><path d="M15 31h8l5-10 8 18 5-8h8M24 47v9m16-9v9M18 56h28"/>',
 tree:'<rect x="23" y="7" width="18" height="14"/><rect x="6" y="43" width="18" height="14"/><rect x="40" y="43" width="18" height="14"/><path d="M32 21v12H15v10m17-10h17v10"/>',
 table:'<rect x="8" y="11" width="48" height="43"/><path d="M8 25h48M8 39h48M24 25v29M40 25v29"/>',
 log:'<path d="M17 8h23l10 11v38H17zM40 8v13h10M25 29h17M25 37h17M25 45h12"/>',
 file:'<path d="M17 8h23l10 11v38H17zM40 8v13h10M25 32h17M25 42h17"/>'
};
// Product logos are the actual vector assets; generic glyphs describe infrastructure/functions only.
const productNodes = {
 'medallion-nix': { kafka:'kafka', flink:'flink', spark:'spark', loader:'spark', pg:'postgresql' },
 'solar-pv': { extract:'python', transform:'python', staging:'postgresql', dwh:'supabase', train:'python', tableau:'tableau', app:'streamlit' },
 'oracle-rac': { asm:'oracle', standby:'oracle' },
 'hadoop-spark': { driver:'spark', spark1:'spark', spark2:'spark', worker1:'spark', worker2:'spark', nn1:'hadoop', nn2:'hadoop', dn1:'hadoop', dn2:'hadoop', jn:'hadoop', zk:'zookeeper' },
 'cdc-iceberg': { pg:'postgresql', deb:'debezium', kafka:'kafka', schema:'confluent', spark:'spark', bronze:'iceberg', silver:'iceberg' },
};
const productHeaders = { 'network-topology':['nixos'], 'oracle-rac':['oracle'], 'oracle-memory':['oracle'], 'hadoop-spark':['hadoop','spark'], 'cdc-iceberg':['postgresql','kafka','iceberg'], tinydb:['rust'], 'medallion-nix':['nixos','flink','spark'], 'solar-pv':['python','postgresql','streamlit'] };
const productUris = new Map();
function productUri(name) {
 if (!productUris.has(name)) productUris.set(name, 'data:image/svg+xml;base64,' + readFileSync(new URL(`../../public/architecture/icons/${name}.svg`, import.meta.url)).toString('base64'));
 return productUris.get(name);
}
for (const d of diagrams) for (const n of d.nodes) n.brand = productNodes[d.id]?.[n.id];
function iconBody(n) {
 if (n.brand) return `<rect width="64" height="64" rx="8" fill="#fff"/><image x="3" y="3" width="58" height="58" href="${productUri(n.brand)}"/>`;
 return `<rect width="64" height="64" fill="${C[n.tone]}"/><g stroke="white" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${icons[n.icon]||icons.server}</g>`;
}
function wrap(str,max) { const out=[];let line='';for(const word of str.split(' ')){if((line+' '+word).length>max&&line){out.push(line);line=word;}else line+=(line?' ':'')+word;}if(line)out.push(line);return out;}
function txt(str,x,y,size=15,color=C.ink,anchor='start',weight=400) {return `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" text-anchor="${anchor}" font-weight="${weight}">${esc(str)}</text>`;}
function renderSvg(d) {
 let s=`<svg xmlns="http://www.w3.org/2000/svg" width="${d.width}" height="${d.height}" viewBox="0 0 ${d.width} ${d.height}" role="img" aria-labelledby="title desc"><title id="title">${esc(d.title)}</title><desc id="desc">${esc(d.subtitle+'; '+d.footnote)}</desc><defs>${Object.entries(C).map(([k,v])=>`<marker id="arrow-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1L9 5L1 9" fill="none" stroke="${v}" stroke-width="1.5"/></marker>`).join('')}</defs><rect width="1600" height="${d.height}" fill="#fff"/><g font-family="'DejaVu Sans',sans-serif">`;
 s+=`<rect x="40" y="43" width="5" height="53" fill="#ed8b23"/>${txt(d.title,65,73,31,C.ink,'start',700)}${txt(d.subtitle,65,105,14,C.muted)}`;
 (productHeaders[d.id] || []).forEach((name,i,all) => {s+=`<image x="${1540-(all.length-i)*65}" y="40" width="60" height="48" href="${productUri(name)}"/>`;});
 for(const z of d.zones){s+=`<rect x="${z.x}" y="${z.y}" width="${z.w}" height="${z.h}" fill="${fills[z.tone]}" stroke="${C[z.tone]}" stroke-width="1.5" ${z.tone==='ink'?'stroke-dasharray="7 5"':''}/><rect x="${z.x}" y="${z.y}" width="28" height="28" fill="${C[z.tone]}"/><path d="M${z.x+7} ${z.y+8}h14v12h-14z" fill="none" stroke="white" stroke-width="1.5"/>${txt(z.label,z.x+40,z.y+20,14,C[z.tone],'start',600)}`;}
 for(const e of d.edges){s+=`<path d="${e.points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' ')}" fill="none" stroke="${C[e.tone]}" stroke-width="1.8" stroke-linejoin="round" ${e.kind.startsWith('control')?'stroke-dasharray="6 5"':''} ${e.kind.endsWith('duplex')?`marker-start="url(#arrow-${e.tone})"`:''} ${e.kind!=='link'?`marker-end="url(#arrow-${e.tone})"`:''}/>`;}
 for(const n of d.nodes){const cx=n.x+n.w/2;s+=`<g><rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" fill="${fills[n.tone]}" stroke="${C[n.tone]}" stroke-opacity=".22" stroke-width="1"/><g transform="translate(${cx-26},${n.y}) scale(.8125)">${iconBody(n)}</g>${txt(n.label,cx,n.y+73,16,C.ink,'middle',600)}${txt(n.detail,cx,n.y+95,12.5,C.muted,'middle')}</g>`;}
 for(const e of d.edges){if(e.label){const x=e.lx,y=e.ly,w=e.label.length*7+18;s+=`<rect x="${x-w/2}" y="${y-14}" width="${w}" height="22" rx="3" fill="#fff"/>${txt(e.label,x,y+1,12.5,C[e.tone],'middle')}`;}}
 for(const n of d.notes){wrap(n.text,Math.floor(n.w/7)).forEach((t,i)=>{s+=txt(t,n.x,n.y+i*19,12.5,C[n.tone]||C.muted);});}
 const ly=d.height-83;s+=`<path d="M40 ${ly-24}H1560" stroke="#dce2e9"/>`;
 s+=`<path d="M45 ${ly}h45" stroke="${C.ink}" stroke-width="2" marker-end="url(#arrow-ink)"/>${txt('Data / request',100,ly+5,13,C.muted)}<path d="M280 ${ly}h45" stroke="${C.control}" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#arrow-control)"/>${txt('Control / metadata',335,ly+5,13,C.muted)}<path d="M545 ${ly}h45" stroke="${C.network}" stroke-width="2"/>${txt('Network link',600,ly+5,13,C.muted)}<path d="M770 ${ly}h45" stroke="${C.database}" stroke-width="2" marker-start="url(#arrow-database)" marker-end="url(#arrow-database)"/>${txt('Two-way exchange',825,ly+5,13,C.muted)}${txt('NGÔ TẤN ĐẠT  /  ARCHITECTURE STUDIES',1550,ly+5,12,C.muted,'end')}`;
 s+=txt(d.footnote,45,d.height-32,13,C.muted);
 return s+'</g></svg>';
}
function cell(id,value,style,x,y,w,h,parent='1') { return `<mxCell id="${id}" value="${esc(value)}" style="${esc(style+';whiteSpace=wrap;html=1;')}" vertex="1" parent="${parent}"><mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry"/></mxCell>`; }
function renderPage(d) {
 const prefix=d.id+'-';const pid=id=>id?prefix+id:'1';
 let s=`<diagram id="${d.id}" name="${esc(d.title)}"><mxGraphModel dx="1600" dy="${d.height}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="${d.height}" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>`;
 s+=cell(prefix+'title',d.title,'text;strokeColor=none;fillColor=none;fontSize=30;fontStyle=1;align=left;fontColor='+C.ink,60,40,1480,50);
 s+=cell(prefix+'subtitle',d.subtitle,'text;strokeColor=none;fillColor=none;fontSize=14;align=left;fontColor='+C.muted,60,90,1480,25);
 (productHeaders[d.id] || []).forEach((name,i,all) => {s+=cell(prefix+'brand-'+name,'',`shape=image;imageAspect=1;image=${productUri(name)};`,1540-(all.length-i)*65,40,60,48);});
 for(const z of d.zones){const p=d.zones.find(p=>p.id===z.parent);s+=cell(pid(z.id),z.label,`swimlane;startSize=32;horizontal=1;rounded=0;fillColor=${fills[z.tone]};swimlaneFillColor=${fills[z.tone]};strokeColor=${C[z.tone]};fontColor=${C[z.tone]};fontSize=14;fontStyle=1;align=left;spacingLeft=12;collapsible=0`,z.x-(p?.x||0),z.y-(p?.y||0),z.w,z.h,pid(z.parent));}
 for(const n of d.nodes){const p=d.zones.find(p=>p.id===n.parent);s+=cell(pid(n.id),'',`group;connectable=1;fillColor=none;strokeColor=none`,n.x-(p?.x||0),n.y-(p?.y||0),n.w,n.h,pid(n.parent));
 const icon=`<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">${iconBody(n)}</svg>`;
 s+=cell(pid(n.id)+'-icon','',`shape=image;imageAspect=0;image=data:image/svg+xml,${encodeURIComponent(icon)};`,n.w/2-26,0,52,52,pid(n.id));
 s+=cell(pid(n.id)+'-label',n.label,`text;strokeColor=none;fillColor=none;fontSize=16;fontStyle=1;fontColor=${C.ink}`,0,58,n.w,23,pid(n.id));
 s+=cell(pid(n.id)+'-detail',n.detail,`text;strokeColor=none;fillColor=none;fontSize=12;fontColor=${C.muted}`,0,82,n.w,20,pid(n.id));}
 d.edges.forEach((e,i)=>{const a=e.points[0],b=e.points.at(-1);const sn=d.nodes.find(n=>n.id===e.from),tn=d.nodes.find(n=>n.id===e.to);const start=[(a[0]-sn.x)/sn.w,(a[1]-sn.y)/sn.h],end=[(b[0]-tn.x)/tn.w,(b[1]-tn.y)/tn.h];
 s+=`<mxCell id="${prefix}edge-${i}" style="edgeStyle=none;rounded=0;html=1;strokeColor=${C[e.tone]};strokeWidth=2;startArrow=${e.kind.endsWith('duplex')?'open':'none'};startFill=0;endArrow=${e.kind==='link'?'none':'open'};endFill=0;dashed=${e.kind.startsWith('control')?1:0};exitX=${start[0]};exitY=${start[1]};exitPerimeter=0;entryX=${end[0]};entryY=${end[1]};entryPerimeter=0;" edge="1" parent="1" source="${pid(e.from)}" target="${pid(e.to)}"><mxGeometry relative="1" as="geometry"><Array as="points">${e.points.slice(1,-1).map(([x,y])=>`<mxPoint x="${x}" y="${y}"/>`).join('')}</Array></mxGeometry></mxCell>`;
 if(e.label)s+=cell(prefix+'edge-label-'+i,e.label,`text;strokeColor=none;fillColor=#ffffff;fontSize=12;fontColor=${C[e.tone]}`,e.lx-e.label.length*3.6-10,e.ly-15,e.label.length*7.2+20,25);
 });
 d.notes.forEach((n,i)=>{s+=cell(prefix+'note-'+i,n.text,`text;strokeColor=none;fillColor=none;align=left;fontSize=12;fontColor=${C[n.tone]||C.muted}`,n.x,n.y-15,n.w,45);});
 s+=cell(prefix+'legend','→ Data / request     - - → Control / metadata     ── Network link     ↔ Two-way exchange',`text;strokeColor=none;fillColor=none;align=left;fontSize=13;fontColor=${C.muted}`,40,d.height-98,1490,30);
 s+=cell(prefix+'footnote',d.footnote,`text;strokeColor=none;fillColor=none;align=left;fontSize=13;fontColor=${C.muted}`,40,d.height-53,1490,35);
 return s+'</root></mxGraphModel></diagram>';
}
const [format,id]=process.argv.slice(2);
// Keep the tinydb source available locally, but exclude it from the public portfolio bundle.
const selected=id==='all'?diagrams.filter(d=>!['tinydb','cdc-iceberg'].includes(d.id)):diagrams.filter(d=>d.id===id);
if(!selected.length||!['svg','drawio','list'].includes(format))throw Error('Expected svg|drawio|list and a diagram id or all');
if(format==='list')console.log(JSON.stringify(diagrams.map(({id,title,subtitle,footnote})=>({id,title,subtitle,footnote}))));
if(format==='svg'){if(selected.length!==1)throw Error('SVG needs one id');console.log(renderSvg(selected[0]));}
if(format==='drawio')console.log(`<mxfile host="app.diagrams.net" modified="2026-09-14T00:00:00.000Z" version="26.0.0">${selected.map(renderPage).join('')}</mxfile>`);
