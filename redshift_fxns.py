############################
###AWS Redshift functions###
############################
import psycopg2

def create_rs_conn(config):
	'''Create and return a redshift connection object, given
	the "config" information'''

    try:
        conn = psycopg2.connect(dbname=config['dbname'], host=config['host'], port=config['port'], user=config['user'], password=config['pwd'])
    except Exception as err:
        print err.code, err
    
    return conn

def get_available_rs_tables(cursor, *args, **kwargs):
	'''Grab a list of all tables in the redshift db'''

    query = """select distinct(tablename) from pg_table_def where schemaname = 'public';"""
    cur = cursor
    cur.execute(query)
    rows = cur.fetchall()
    header = [colnames[0] for colnames in cur.description]
    return header, rows

def run_rs_query(cursor, query):
	'''run a readshift query'''
    cur = cursor
    cur.execute(query)
    rows = cur.fetchall()
    header = [colnames[0] for colnames in cur.description]
    return header, rows

def close_rs_conn(cursor, connection):
	'''Close Redshift connection (with exception handling)'''
    try:
        cursor.close()
        connection.close()
    except Exception as err:
        print err.code, err
    
    return "Redshift connection successfully closed!"