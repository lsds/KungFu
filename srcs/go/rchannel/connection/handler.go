package connection

type Handler interface {
	Handle(conn Connection) (int, error)
}
