package rchannel

// Channel is a remote channel
type Channel struct {
	name string
	conn *Connection
}

func newChannel(name string, conn *Connection) *Channel {
	return &Channel{
		name: name,
		conn: conn,
	}
}

// Send sends a message to remove channel
func (c *Channel) Send(m Message) error {
	return c.conn.send(c.name, m)
}
